import json
import logging
import os
import threading
import time
from dataclasses import dataclass, astuple
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

import phoenixsystems.sem.metersim as metersim
from cognit.device_runtime import DeviceRuntime
from home_energy_management.device_simulators.device_utils import DeviceUserApi
from home_energy_management.device_simulators.heating import HeatingPreferences
from home_energy_management.device_simulators.electric_vehicle import EVDeparturePlans


@dataclass
class AlgoPredictParams:
    timestamp: float
    s3_parameters: str
    besmart_parameters: str
    home_model_parameters: str
    storage_parameters: str
    ev_battery_parameters_per_id: str
    heating_parameters: str
    user_preferences: str


@dataclass
class AlgoTrainParams:
    train_parameters: str
    s3_parameters: str
    besmart_parameters: str
    home_model_parameters: str
    storage_parameters: str
    ev_battery_parameters_per_id: str
    heating_parameters: str
    user_preferences: str


class UserApp:
    sem_id: int
    start_date: datetime
    device_runtime: DeviceRuntime  # Cognit Serverless Runtime
    metrology: metersim.Metersim  # Metrology
    heating_user_preferences: HeatingPreferences
    ev_departure_plans: dict[str, EVDeparturePlans]
    cycle_time: int
    cycle_train_time: int
    model_parameters: dict[str, float]
    train_parameters: dict[str, Any]
    s3_parameters: dict[str, str]
    besmart_parameters: dict[str, Any]
    user_preferences: dict[str, Any]

    # Offloaded functions
    training_algo: Callable
    decision_algo: Callable

    # Devices
    pv: DeviceUserApi
    energy_storage: DeviceUserApi
    electric_vehicle_per_id: Mapping[str, DeviceUserApi]
    heating: DeviceUserApi
    temp_outside_sensor: DeviceUserApi

    # Utils
    shutdown_flag: bool
    app_thread: threading.Thread
    use_cognit: bool
    use_model: bool
    start_time: float
    cond: threading.Condition
    app_logger: logging.Logger
    cognit_logger: logging.Logger
    global_logger: logging.Logger = None
    training_state_cb: Callable

    # Registers
    last_predict_run: float = 0.0
    last_train_run: float = 0.0
    last_active_plus: int = 0
    last_active_minus: int = 0
    last_pv_energy: float = 0.0
    last_storage_charge_level: float = 0.0
    last_ev_battery_charge_level_per_id: dict[str, float] = {}

    def __init__(
        self,
        sem_id: int,
        start_date: datetime,
        metrology: metersim.Metersim,
        decision_algo: Callable,
        model_parameters: dict[str, float],
        pv: DeviceUserApi,
        energy_storage: DeviceUserApi,
        electric_vehicle_per_id: Mapping[str, DeviceUserApi],
        heating: DeviceUserApi,
        temp_outside_sensor: DeviceUserApi,
        cycle: int,
        cycle_train: int,
        heating_user_preferences: HeatingPreferences,
        ev_departure_plans: dict[str, EVDeparturePlans],
        user_preferences: dict[str, Any],
        besmart_parameters: dict[str, Any],
        use_cognit: bool = True,
        reqs_init: dict[str, Any] = None,
        use_model: bool = True,
        training_algo: Callable = None,
        s3_parameters: dict[str, str] = None,
        train_parameters: dict[str, Any] = None,
        training_state_cb: Callable = None,
    ):
        self.sem_id = sem_id
        self.start_date = start_date
        self.metrology = metrology
        self.decision_algo = decision_algo
        self.model_parameters = model_parameters
        self.pv = pv
        self.energy_storage = energy_storage
        self.electric_vehicle_per_id = electric_vehicle_per_id
        self.last_ev_battery_charge_level_per_id = {ev_id: 0.0 for ev_id in self.electric_vehicle_per_id.keys()}
        self.heating = heating
        self.temp_outside_sensor = temp_outside_sensor
        self.cycle_time = cycle
        self.cycle_train_time = cycle_train
        self.heating_user_preferences = heating_user_preferences
        self.ev_departure_plans = ev_departure_plans
        self.use_cognit = use_cognit
        self.besmart_parameters = besmart_parameters
        self.user_preferences = user_preferences

        self.use_model = use_model
        self.training_algo = training_algo
        self.s3_parameters = s3_parameters
        self.train_parameters = train_parameters

        self.shutdown_flag = False
        self.cond = threading.Condition()

        app_log_handler = logging.FileHandler(f"log/{os.getpid()}/{self.sem_id}/user_app.log")
        app_log_formatter = logging.Formatter("")
        app_log_handler.setFormatter(app_log_formatter)
        self.app_logger = logging.Logger("user_app")
        self.app_logger.addHandler(app_log_handler)
        app_log_handler.setLevel(logging.INFO)

        self.training_state_cb = training_state_cb

        if self.use_cognit:
            self._init_cognit_runtime(reqs_init)

        self.app_thread = threading.Thread(target=self._app_loop)

    def _init_cognit_runtime(self, reqs_init: dict[str, Any]) -> None:
        self.cognit_logger = logging.getLogger("cognit-logger")
        self.cognit_logger.handlers.clear()
        handler = logging.FileHandler(f"log/{os.getpid()}/{self.sem_id}/cognit.log")
        formatter = logging.Formatter(
            fmt="[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.cognit_logger.addHandler(handler)

        self.global_logger = logging.Logger("global-logger")
        handler = logging.FileHandler("log/cognit.log")
        formatter = logging.Formatter(
            fmt="[%(process)d][%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.global_logger.addHandler(handler)

        self.device_runtime = DeviceRuntime("cognit.yml")
        self.device_runtime.init(reqs_init)

        self.cognit_logger.info("Runtime should be ready now!")

    def _get_sem_time(self) -> int:
        return self.metrology.get_time_utc()

    def offload_predict_now(self):
        with self.cond:
            self._offload_predict()
            self.cond.notify_all()

    def offload_train_now(self):
        with self.cond:
            self._offload_train()
            self.cond.notify_all()

    def set_cycle_length(self, cycle: int):
        with self.cond:
            self.cycle_time = cycle
            self._offload_train()
            self.cond.notify_all()

    def set_cycle_train_length(self, cycle_train: int):
        with self.cond:
            self.cycle_train_time = cycle_train
            self.cond.notify_all()

    def _update_predict_algo_input(self, now: float) -> AlgoPredictParams:
        self.last_predict_run = now
        next_timestamp = self.start_date + timedelta(seconds=self.metrology.get_uptime() + self.cycle_time)

        storage_parameters = self.energy_storage.get_info()
        ev_parameters_per_id = {}
        for ev_id, electric_vehicle in self.electric_vehicle_per_id.items():
            ev_parameters = electric_vehicle.get_info()
            ev_parameters["time_until_charged"] = self.ev_departure_plans[ev_id].get_time_until_departure()
            ev_parameters_per_id[ev_id] = ev_parameters
        heating_parameters = self.heating.get_info()
        heating_parameters["preferred_temp"] = self.heating_user_preferences.get_temp()
        self.user_preferences["cycle_timedelta_s"] = self.cycle_time

        energy = self.metrology.get_energy_total()
        self.last_active_plus = energy.active_plus
        self.last_active_minus = energy.active_minus
        self.last_pv_energy = self.pv.get_info()["energy_produced"]
        self.last_storage_charge_level = storage_parameters["curr_charge_level"]
        for ev_id in self.last_ev_battery_charge_level_per_id.keys():
            self.last_ev_battery_charge_level_per_id[ev_id] = ev_parameters_per_id[ev_id]["curr_charge_level"]

        algo_input = AlgoPredictParams(
            next_timestamp.timestamp(),
            json.dumps(self.s3_parameters),
            json.dumps(self.besmart_parameters),
            json.dumps(self.model_parameters),
            json.dumps(storage_parameters),
            json.dumps(ev_parameters_per_id),
            json.dumps(heating_parameters),
            json.dumps(self.user_preferences),
        )
        return algo_input

    def _update_training_algo_input(self, now: float) -> AlgoTrainParams:
        self.last_predict_run = now
        last_timestamp = self.start_date + timedelta(seconds=self.metrology.get_uptime() - self.cycle_time)
        first_timestamp = last_timestamp - timedelta(days=self.train_parameters["history_timedelta_days"])
        self.besmart_parameters["since"] = first_timestamp.timestamp()
        self.besmart_parameters["till"] = last_timestamp.timestamp()
        ev_parameters_per_id = {}
        for ev_id, electric_vehicle in self.electric_vehicle_per_id.items():
            ev_parameters_per_id[ev_id] = electric_vehicle.get_info()
        self.user_preferences["cycle_timedelta_s"] = self.cycle_time

        algo_input = AlgoTrainParams(
            json.dumps(self.train_parameters),
            json.dumps(self.s3_parameters),
            json.dumps(self.besmart_parameters),
            json.dumps(self.model_parameters),
            json.dumps(self.energy_storage.get_info()),
            json.dumps(ev_parameters_per_id),
            json.dumps(self.heating.get_info()),
            json.dumps(self.user_preferences),
        )
        return algo_input

    def _execute_algo_response(self, algo_res: str):
        (conf_temp, storage_params, ev_params_per_id, *_) = algo_res
        self.energy_storage.set_params(json.loads(storage_params))
        ev_params_per_id = json.loads(ev_params_per_id)
        for ev_id, ev_params in ev_params_per_id.items():
            self.electric_vehicle_per_id[ev_id].set_params(ev_params)
        self.heating.set_params(
            {
                "optimal_temp": conf_temp,
            }
        )

    def _run_algo(self, algo_function: Callable, algo_input: AlgoPredictParams | AlgoTrainParams) -> Any:
        if not self.use_cognit:
            res = algo_function(*astuple(algo_input))
        else:
            try:
                ret = self.device_runtime.call(algo_function, *astuple(algo_input))
                res = ret.res
                self.app_logger.info(f"\nRuntime result: {res}")
                if ret.err:
                    self.app_logger.error(f"\nError during offloading: {ret.err}")
                self.global_logger.info("Offload OK")
            except:
                self.global_logger.error("Offload ERROR")
                raise
        return res

    def start(self):
        self.start_time = self._get_sem_time()
        self.app_thread.start()

    def destroy(self):
        self.shutdown_flag = True
        with self.cond:
            self.cond.notify_all()
        self.app_thread.join()

    def _offload_predict(self):
        now = self._get_sem_time()
        algo_input = self._update_predict_algo_input(now)
        algo_res = self._run_algo(self.decision_algo, algo_input)

        self.app_logger.info("\n\x1b[2J\x1b[H")
        self.app_logger.info(f"{self.start_date + timedelta(seconds=self.metrology.get_uptime())}")
        self.app_logger.info("\n\tINPUT")
        model_parameters = json.loads(algo_input.home_model_parameters)
        self.app_logger.info(
            f"Model parameters: \n\t- heat capacity (J/K): {model_parameters['heat_capacity']}, "
            f"\n\t- heating temperature window (K): {model_parameters['temp_window']}, "
            f"\n\t- heating coefficient: {model_parameters['heating_coefficient']}, "
            f"\n\t- heat loss coefficient (W/K): {model_parameters['heat_loss_coefficient']},"
            f"\n\t- minimal temperature setting (°C): {model_parameters['min_temp_setting']},"
            f"\n\t- maximal temperature setting (°C): {model_parameters['max_temp_setting']}."
        )
        storage_parameters = json.loads(algo_input.storage_parameters)
        self.app_logger.info(
            f"Storage parameters: \n\t- max capacity (kWh): {storage_parameters['max_capacity']}, "
            f"\n\t- nominal power (kW): {storage_parameters['nominal_power']}, "
            f"\n\t- efficiency: {storage_parameters['efficiency']}."
        )
        ev_battery_parameters_per_id = json.loads(algo_input.ev_battery_parameters_per_id)
        for ev_id, ev_battery_parameters in ev_battery_parameters_per_id.items():
            time_until_ev_charged = ev_battery_parameters["time_until_charged"]
            self.app_logger.info(
                f"EV (id: {ev_id}) battery parameters:"
                f"\n\t- max capacity (kWh): {ev_battery_parameters['max_capacity']}, "
                f"\n\t- is available: {ev_battery_parameters['is_available']}, "
                f"\n\t- departure SOC (%): {ev_battery_parameters['driving_charge_level']}, "
                f"\n\t- time until charged (h): "
                f"{round(time_until_ev_charged / 3600, 2) if time_until_ev_charged > 0 else 'uknown'}, "
                f"\n\t- nominal power (kW): {ev_battery_parameters['nominal_power']}, "
                f"\n\t- efficiency: {ev_battery_parameters['efficiency']}."
            )
        temperature_inside = json.loads(algo_input.heating_parameters)["curr_temp"]
        preferred_temperature = json.loads(algo_input.heating_parameters)["preferred_temp"]
        self.app_logger.info(f"Inside temperature (°C): {round(temperature_inside, 2)}")
        self.app_logger.info(f"Preferred temperature (°C): {round(preferred_temperature, 2)}")
        self.app_logger.info(f"Current storage SOC (%): {round(storage_parameters['curr_charge_level'], 2)}")
        for ev_id, ev_battery_parameters in ev_battery_parameters_per_id.items():
            self.app_logger.info(
                f"Current EV (id: {ev_id}) battery SOC (%): {round(ev_battery_parameters['curr_charge_level'], 2)}"
            )

        if algo_res is not None:
            self._execute_algo_response(algo_res)
            self.app_logger.info("\n\tOUTPUT")
            self.app_logger.info(f"Configuration of temperature (°C): {round(algo_res[0], 2)}")
            self.app_logger.info(f"Configuration of storage: {json.dumps(json.loads(algo_res[1]), indent=4)}")
            ev_battery_conf = json.loads(algo_res[2])
            if len(ev_battery_conf) > 0:
                self.app_logger.info("Configuration of EV battery:")
                for ev_id in ev_battery_conf:
                    self.app_logger.info(f"\t{ev_id}: {json.dumps(ev_battery_conf[ev_id], indent=4)}")
        else:
            self.app_logger.warning("Decision algorithm call failed")

    def _offload_train(self):
        now = self._get_sem_time()
        algo_input = self._update_training_algo_input(now)

        self.app_logger.info("\n\x1b[2J\x1b[H")
        self.app_logger.info(f"{self.start_date + timedelta(seconds=self.metrology.get_uptime())}")
        self.app_logger.info("Training parameters:")
        self.app_logger.info(f"{json.dumps(json.loads(algo_input.train_parameters), indent=4)}")

        if self.training_state_cb is not None:
            self.training_state_cb(1)
        start_time = time.time()
        algo_res = self._run_algo(self.training_algo, algo_input)
        end_time = time.time()
        if self.training_state_cb is not None:
            self.training_state_cb(0)

        if algo_res is not None:
            self.model_parameters["state_range"] = json.loads(algo_res)
            self.app_logger.info(f"\nTraining completed in: {(end_time - start_time):.2f} seconds")
            self.app_logger.info(f"Model saved in s3 bucket")
        else:
            self.app_logger.warning("Training decision model call failed")

    def _app_loop(self):
        now = self._get_sem_time()
        self.last_predict_run = now
        self.last_train_run = now

        with self.cond:
            while not self.shutdown_flag:
                now = self._get_sem_time()

                training_performed = False
                if self.use_model and now >= self.last_train_run + self.cycle_train_time:
                    self._offload_train()
                    training_performed = True

                if training_performed or now >= self.last_predict_run + self.cycle_time:
                    self._offload_predict()

                self.cond.wait(1)  # Sleep 1 second
