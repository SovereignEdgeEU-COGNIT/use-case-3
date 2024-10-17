import json
from dataclasses import dataclass, astuple
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping
import os

import numpy as np
import pandas as pd
import phoenixsystems.sem.metersim as metersim
import torch
from cognit import device_runtime

from home_energy_management.device_simulators.device_utils import DeviceUserApi
from home_energy_management.device_simulators.heating import HeatingPreferences
from home_energy_management.device_simulators.electric_vehicle import EVDeparturePlans


REQS_INIT = {
      "FLAVOUR": "EnergyV2__16GB_4CPU",
      "MIN_ENERGY_RENEWABLE_USAGE": 50,
}

@dataclass
class AlgoPredictParams:
    timestamp: float
    trained_model: bytes
    home_model_parameters: dict[str, float]
    storage_parameters: dict[str, float]
    ev_battery_parameters: dict[str, float]
    heating_parameters_per_room: list[dict[str, Any]]
    energy_pv_produced_pred: float
    uncontrolled_energy_consumption_pred: float
    temp_outside_pred: float
    cycle_timedelta_s: int

@dataclass
class AlgoTrainParams:
    train_parameters: dict[str, Any]
    home_model_parameters: dict[str, float]
    storage_parameters: dict[str, float]
    ev_battery_parameters: dict[str, float]
    heating_parameters: dict[str, Any]
    cycle_timedelta_s: int
    timestamps_hour: np.ndarray[int]
    pv_generation_train: np.ndarray[float]
    pv_generation_pred_train: np.ndarray[float]
    uncontrolled_consumption_train: np.ndarray[float]
    uncontrolled_consumption_pred_train: np.ndarray[float]
    temp_outside_train: np.ndarray[float]
    temp_outside_pred_train: np.ndarray[float]

@dataclass
class TimeSeries:
    time: np.ndarray[int]
    value: np.ndarray[float]

    @staticmethod
    def from_csv(path_to_csv) -> 'TimeSeries':
        df = pd.read_csv(path_to_csv)
        return TimeSeries(df['unix_timestamp'].values,
                          df['value'].values)

    def get_value_by_timestamp(self, timestamp: datetime) -> float:
        timestamp = self.__timestamp_hour_rounder(timestamp)
        try:
            index = np.where(self.time == timestamp.timestamp())[0][0]
            return self.value[index]
        except IndexError:
            return np.nan

    def get_values_between_timestamps(self,
                                      start_timestamp: datetime,
                                      end_timestamp: datetime) -> np.ndarray[float]:
        start_timestamp = self.__timestamp_hour_rounder(start_timestamp)
        end_timestamp = self.__timestamp_hour_rounder(end_timestamp)
        indexes = np.where(np.logical_and(self.time >= start_timestamp.timestamp(),
                                          self.time < end_timestamp.timestamp()))[0]
        return self.value[indexes]

    def get_times_between_timestamps(self,
                                     start_timestamp: datetime,
                                     end_timestamp: datetime) -> np.ndarray[int]:
        start_timestamp = self.__timestamp_hour_rounder(start_timestamp)
        end_timestamp = self.__timestamp_hour_rounder(end_timestamp)
        indexes = np.where(np.logical_and(self.time >= start_timestamp.timestamp(),
                                          self.time < end_timestamp.timestamp()))[0]
        return self.time[indexes]

    @staticmethod
    def __timestamp_hour_rounder(t: datetime) -> datetime:
        # Rounds to nearest hour by adding a timedelta hour if minute >= 30
        return t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(hours=t.minute // 30)

class UserApp:
    start_date: datetime
    runtime: device_runtime.DeviceRuntime  # Cognit Serverless Runtime
    metrology: metersim.Metersim  # Metrology
    heating_user_preferences: dict[str, HeatingPreferences]
    ev_departure_plans: EVDeparturePlans
    cycle_time: int
    speedup: int
    num_cycles_retrain: int
    model_parameters: dict[str, float]
    train_parameters: dict[str, Any]
    model_path: str

    # Offloaded functions
    training_algo: Callable
    decision_algo: Callable

    # Devices
    pv: DeviceUserApi
    energy_storage: DeviceUserApi
    electric_vehicle: DeviceUserApi
    room_heating: Mapping[str, DeviceUserApi]
    temp_outside_sensor: DeviceUserApi

    # Utils
    shutdown_flag: bool
    app_thread: threading.Thread
    use_cognit: bool
    cognit_timeout: int
    start_time: float
    cond: threading.Condition
    app_logger: logging.Logger
    cognit_logger: logging.Logger

    # Registers
    last_algo_run: float = 0.0
    last_active_plus: int = 0
    last_active_minus: int = 0
    last_pv_energy: float = 0.0
    last_storage_charge_level: float = 0.0
    last_ev_battery_charge_level: float = 0.0

    # Time series
    pv_production_series: TimeSeries = TimeSeries(np.array([]), np.array([]))
    pv_production_pred_series: TimeSeries = TimeSeries(np.array([]), np.array([]))
    consumption_pred_series: TimeSeries = TimeSeries(np.array([]), np.array([]))
    temp_outside_series: TimeSeries = TimeSeries(np.array([]), np.array([]))
    temp_outside_pred_series: TimeSeries = TimeSeries(np.array([]), np.array([]))

    def __init__(
            self,
            start_date: datetime,
            metrology: metersim.Metersim,
            decision_algo: Callable,
            training_algo: Callable,
            model_path: str,
            model_parameters: dict[str, float],
            train_parameters: dict[str, Any],
            pv: DeviceUserApi,
            energy_storage: DeviceUserApi,
            electric_vehicle: DeviceUserApi,
            room_heating: Mapping[str, DeviceUserApi],
            temp_outside_sensor: DeviceUserApi,
            speedup: int,
            cycle: int,
            num_cycles_retrain: int,
            heating_user_preferences: dict[str, HeatingPreferences],
            ev_departure_plans: EVDeparturePlans,
            pv_production_series: TimeSeries,
            pv_production_pred_series: TimeSeries,
            consumption_series: TimeSeries,
            consumption_pred_series: TimeSeries,
            temp_outside_series: TimeSeries,
            temp_outside_pred_series: TimeSeries,
            use_cognit: bool = True,
            cognit_timeout: int = 3,
    ) -> None:
        self.start_date = start_date
        self.metrology = metrology
        self.decision_algo = decision_algo
        self.training_algo = training_algo
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.train_parameters = train_parameters
        self.pv = pv
        self.energy_storage = energy_storage
        self.electric_vehicle = electric_vehicle
        self.room_heating = room_heating
        self.temp_outside_sensor = temp_outside_sensor
        self.use_cognit = use_cognit
        self.cognit_timeout = cognit_timeout
        self.speedup = speedup
        self.cycle_time = cycle
        self.num_cycles_retrain = num_cycles_retrain
        self.heating_user_preferences = heating_user_preferences
        self.ev_departure_plans = ev_departure_plans

        self.pv_production_series = pv_production_series
        self.pv_production_pred_series = pv_production_pred_series
        self.consumption_series = consumption_series
        self.consumption_pred_series = consumption_pred_series
        self.temp_outside_series = temp_outside_series
        self.temp_outside_pred_series = temp_outside_pred_series

        self.shutdown_flag = False
        self.cond = threading.Condition()

        app_log_handler = logging.FileHandler("user_app.log")
        app_log_formatter = logging.Formatter("")
        app_log_handler.setFormatter(app_log_formatter)
        self.app_logger = logging.Logger("user_app")
        self.app_logger.addHandler(app_log_handler)
        app_log_handler.setLevel(logging.INFO)

        if self.use_cognit:
            self.init_cognit_runtime()

        self.app_thread = threading.Thread(target=self.app_loop)

    def init_cognit_runtime(self) -> None:
        self.cognit_logger = logging.getLogger("cognit-logger")
        self.cognit_logger.handlers.clear()
        pid = os.getpid()
        handler = logging.FileHandler(f"log/{pid}.log")
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

        self.runtime = device_runtime.DeviceRuntime("cognit.yml")
        self.runtime.init(REQS_INIT)

        self.cognit_logger.info("Runtime should be ready now!")

    def set_heating_user_preferences(self, room: str, pref: HeatingPreferences):
        self.heating_user_preferences[room] = pref

    def update_slr_preferences(self, green_energy_perc: int):
        pass

    def offload_now(self):
        with self.cond:
            self.offload_predict()
            self.cond.notify_all()

    def set_cycle_length(self, cycle: int):
        with self.cond:
            self.cycle_time = cycle
            self.cond.notify_all()

    def set_speedup(self, speedup: int):
        with self.cond:
            self.speedup = speedup
            self.cond.notify_all()

    def update_predict_algo_input(self, now: float) -> AlgoPredictParams:
        self.last_algo_run = now
        next_timestamp = self.start_date + timedelta(seconds=self.metrology.get_uptime() + self.cycle_time)

        storage_parameters = self.energy_storage.get_info()
        ev_parameters = self.electric_vehicle.get_info()
        ev_parameters["time_until_charged"] = self.ev_departure_plans.get_time_until_departure()
        room_heating_params_list = []
        for room, value in self.heating_user_preferences.items():
            params = self.room_heating[room].get_info()
            params["preferred_temp"] = value.get_temp()
            room_heating_params_list.append(params)

        uncontrolled_consumption_pred = self.consumption_pred_series.get_value_by_timestamp(next_timestamp)
        energy_pv_produced_pred = self.pv_production_pred_series.get_value_by_timestamp(next_timestamp)
        temp_outside_pred = self.temp_outside_pred_series.get_value_by_timestamp(next_timestamp)

        energy = self.metrology.get_energy_total()
        self.last_active_plus = energy.active_plus
        self.last_active_minus = energy.active_minus
        self.last_pv_energy = self.pv.get_info()["energy_produced"]
        self.last_storage_charge_level = storage_parameters["curr_charge_level"]
        self.last_ev_battery_charge_level = ev_parameters["curr_charge_level"]

        with open(self.model_path, mode='rb') as file:
            bytes_with_model = file.read()

        algo_input = AlgoPredictParams(
            next_timestamp.timestamp(),
            bytes_with_model,
            self.model_parameters,
            storage_parameters,
            ev_parameters,
            room_heating_params_list,
            energy_pv_produced_pred,
            uncontrolled_consumption_pred,
            temp_outside_pred,
            self.cycle_time
        )
        return algo_input

    def update_training_algo_input(self, now: float) -> AlgoTrainParams:
        self.last_algo_run = now
        last_timestamp = self.start_date + timedelta(seconds=self.metrology.get_uptime() - self.cycle_time)
        first_timestamp = last_timestamp - timedelta(days=self.train_parameters["data_timedelta_days"])
        timestamps = self.pv_production_series.get_times_between_timestamps(first_timestamp, last_timestamp)
        get_hour = lambda x: datetime.fromtimestamp(x).hour
        timestamp_hours = np.vectorize(get_hour)(timestamps)

        room_heating_params_list = []
        for room, value in self.heating_user_preferences.items():
            params = self.room_heating[room].get_info()
            room_heating_params_list.append(params)

        algo_input = AlgoTrainParams(
            self.train_parameters,
            self.model_parameters,
            self.energy_storage.get_info(),
            self.electric_vehicle.get_info(),
            room_heating_params_list[0],
            self.cycle_time,
            timestamp_hours,
            self.pv_production_series.get_values_between_timestamps(first_timestamp, last_timestamp),
            self.pv_production_pred_series.get_values_between_timestamps(first_timestamp, last_timestamp),
            self.consumption_series.get_values_between_timestamps(first_timestamp, last_timestamp),
            self.consumption_pred_series.get_values_between_timestamps(first_timestamp, last_timestamp),
            self.temp_outside_series.get_values_between_timestamps(first_timestamp, last_timestamp),
            self.temp_outside_pred_series.get_values_between_timestamps(first_timestamp, last_timestamp),
        )
        return algo_input

    def execute_algo_response(self, algo_res: Any):
        (
            conf_temp_per_room,
            storage_params,
            ev_params,
            *_
        ) = algo_res
        self.energy_storage.set_params(storage_params)
        self.electric_vehicle.set_params(ev_params)
        for key, value in self.room_heating.items():
            value.set_params(
                {
                    "optimal_temp": conf_temp_per_room[key],
                }
            )

    def run_algo(self, algo_function: Callable, algo_input: AlgoPredictParams | AlgoTrainParams) -> Any:
        ret = None
        if not self.use_cognit:
            ret = algo_function(*astuple(algo_input))
        else:
            try:
                return_code, ret = self.runtime.call(algo_function, *astuple(algo_input))
                self.global_logger.info("OK")
            except:
                self.global_logger.error("ERROR")
        return ret

    def start(self):
        self.start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.app_thread.start()

    def destroy(self):
        self.shutdown_flag = True
        with self.cond:
            self.cond.notify_all()
        self.app_thread.join()

    def offload_predict(self):
        now = time.clock_gettime(time.CLOCK_MONOTONIC)
        algo_input = self.update_predict_algo_input(now)
        algo_res = self.run_algo(self.decision_algo, algo_input)

        self.app_logger.info("\n\x1B[2J\x1B[H")
        self.app_logger.info(f"{self.start_date + timedelta(seconds=self.metrology.get_uptime())}")
        self.app_logger.info("\n\tINPUT")
        model_parameters = algo_input.home_model_parameters
        self.app_logger.info(
            f"Model parameters: \n\t- heat capacity (J/K): {model_parameters['heat_capacity']}, "
            f"\n\t- heating delta temperature (K): {model_parameters['heating_delta_temperature']}, "
            f"\n\t- heating coefficient: {model_parameters['heating_coefficient']}, "
            f"\n\t- heat loss coefficient (W/K): {model_parameters['heat_loss_coefficient']},"
            f"\n\t- minimal temperature setting (°C): {model_parameters['min_temp_setting']},"
            f"\n\t- maximal temperature setting (°C): {model_parameters['max_temp_setting']}."
        )
        storage_parameters = algo_input.storage_parameters
        self.app_logger.info(
            f"Storage parameters: \n\t- max capacity (kWh): {storage_parameters['max_capacity']}, "
            f"\n\t- nominal power (kW): {storage_parameters['nominal_power']}, "
            f"\n\t- efficiency: {storage_parameters['efficiency']}."
        )
        ev_battery_parameters = algo_input.ev_battery_parameters
        time_until_ev_charged = ev_battery_parameters['time_until_charged']
        self.app_logger.info(
            f"EV battery parameters: \n\t- max capacity (kWh): {ev_battery_parameters['max_capacity']}, "
            f"\n\t- is available: {ev_battery_parameters['is_available']}, "
            f"\n\t- departure SOC (%): {ev_battery_parameters['driving_charge_level']}, "
            f"\n\t- time until charged (h): "
            f"{round(time_until_ev_charged / 3600, 2) if time_until_ev_charged > 0 else 'uknown'}, "
            f"\n\t- nominal power (kW): {ev_battery_parameters['nominal_power']}, "
            f"\n\t- efficiency: {ev_battery_parameters['efficiency']}."
        )
        self.app_logger.info(
            f"Prediction of uncontrolled energy consumption (kWh): "
            f"{round(algo_input.uncontrolled_energy_consumption_pred, 2)}"
        )
        self.app_logger.info(
            f"Prediction of energy PV production (kWh): {round(algo_input.energy_pv_produced_pred, 2)}"
        )
        temperature_inside = np.mean(np.array([room["curr_temp"] for room in algo_input.heating_parameters_per_room]))
        preferred_temperature = np.mean(np.array([room["preferred_temp"]
                                                  for room in algo_input.heating_parameters_per_room]))
        self.app_logger.info(f"Inside temperature (°C): {round(temperature_inside, 2)}")
        self.app_logger.info(f"Preferred temperature (°C): {round(preferred_temperature, 2)}")
        self.app_logger.info(f"Prediction of outside temperature (°C): {algo_input.temp_outside_pred}")
        self.app_logger.info(f"Current storage SOC (%): {round(storage_parameters['curr_charge_level'], 2)}")
        self.app_logger.info(f"Current EV battery SOC (%): {round(ev_battery_parameters['curr_charge_level'], 2)}")

        if algo_res is not None:
            self.execute_algo_response(algo_res)
            self.app_logger.info("\n\tOUTPUT")
            temperature_setting = np.mean(np.array([v for v in algo_res[0].values()]))
            self.app_logger.info(
                f"Configuration of temperature (°C): {round(temperature_setting, 2)}"
            )
            self.app_logger.info(f"Configuration of storage: {algo_res[1]}")
            self.app_logger.info(f"Configuration of EV battery: {algo_res[2]}")
        else:
            self.app_logger.warning("Decision algorithm call failed")

    def offload_train(self):
        now = time.clock_gettime(time.CLOCK_MONOTONIC)
        algo_input = self.update_training_algo_input(now)

        self.app_logger.info("\n\x1B[2J\x1B[H")
        self.app_logger.info(f"{self.start_date + timedelta(seconds=self.metrology.get_uptime())}")
        self.app_logger.info("Training parameters:")
        self.app_logger.info(f"{json.dumps(algo_input.train_parameters, indent=4)}")

        start_time = time.time()
        algo_res = self.run_algo(self.training_algo, algo_input)
        end_time = time.time()
        self.app_logger.info(f"\nTraining completed in: {(end_time - start_time):.2f} seconds")

        if algo_res is not None:
            model_scripted = torch.jit.script(algo_res)
            model_scripted.save(self.model_path)
            self.app_logger.info(f"Model saved in {self.model_path}")
        else:
            self.app_logger.warning("Training decision model call failed")

    def app_loop(self):
        self.last_algo_run = time.clock_gettime(time.CLOCK_MONOTONIC)
        counter = 0

        with self.cond:
            while not self.shutdown_flag:
                slept = False

                while (
                        time.clock_gettime(time.CLOCK_MONOTONIC)
                        < (self.cycle_time / self.speedup + self.last_algo_run)
                        and not self.shutdown_flag
                ) or not slept:
                    sleep_time = max(
                        self.cycle_time / self.speedup
                        + self.last_algo_run
                        - time.clock_gettime(time.CLOCK_MONOTONIC),
                        0.001,  # Sleep 1 ms to allow other threads to acquire cond
                    )

                    self.cond.wait(sleep_time)
                    slept = True

                if counter % self.num_cycles_retrain == 0:
                    old_speedup = self.speedup
                    self.set_speedup(1)
                    self.metrology.set_speedup(1)
                    self.offload_train()
                    self.set_speedup(old_speedup)
                    self.metrology.set_speedup(old_speedup)
                    counter = 0
                counter += 1
                self.offload_predict()
