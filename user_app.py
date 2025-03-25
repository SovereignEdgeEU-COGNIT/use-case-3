from dataclasses import dataclass, astuple
import logging
import math
import time
import threading
from datetime import timedelta
from typing import Any, Callable, Mapping
import sys

from cognit import device_runtime

import phoenixsystems.sem.metersim as metersim

from home_energy_management.device_simulators.device_utils import DeviceUserApi
from home_energy_management.device_simulators.heating import HeatingPreferences
from home_energy_management.device_simulators.electric_vehicle import EVDeparturePlans


REQS_INIT = {
      "FLAVOUR": "EnergyV2",
      "MIN_ENERGY_RENEWABLE_USAGE": 50,
}

@dataclass
class AlgoParams:
    model_parameters: dict[str, float]
    step_timedelta_s: int
    storage_parameters: dict[str, float]
    ev_battery_parameters: dict[str, float]
    room_heating_params_list: list[dict[str, Any]]
    energy_drawn_from_grid: float
    energy_returned_to_grid: float
    energy_pv_produced: float
    temp_outdoor: float
    charge_level_of_storage: float
    prev_charge_level_of_storage: float
    charge_level_of_ev_battery: float
    prev_charge_level_of_ev_battery: float
    heating_status_per_room: dict[str, list[bool]]
    temp_per_room: dict[str, float]


class UserApp:
    metrology: metersim.Metersim  # Metrology
    runtime: device_runtime.DeviceRuntime  # Cognit Serverless Runtime
    heating_user_preferences: dict[str, HeatingPreferences]
    ev_departure_plans: EVDeparturePlans
    cycle_time: int
    speedup: int
    model_parameters: dict[str, float]

    # Decision algorithm
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

    def __init__(
            self,
            metrology: metersim.Metersim,
            decision_algo: Callable,
            model_parameters: dict[str, float],
            pv: DeviceUserApi,
            energy_storage: DeviceUserApi,
            electric_vehicle: DeviceUserApi,
            room_heating: Mapping[str, DeviceUserApi],
            temp_outside_sensor: DeviceUserApi,
            speedup: int,
            cycle: int,
            heating_user_preferences: dict[str, HeatingPreferences],
            ev_departure_plans: EVDeparturePlans,
            use_cognit: bool = True,
            cognit_timeout: int = 3,
    ) -> None:
        self.metrology = metrology
        self.decision_algo = decision_algo
        self.pv = pv
        self.energy_storage = energy_storage
        self.electric_vehicle = electric_vehicle
        self.room_heating = room_heating
        self.temp_outside_sensor = temp_outside_sensor
        self.use_cognit = use_cognit
        self.cognit_timeout = cognit_timeout
        self.speedup = speedup
        self.cycle_time = cycle
        self.heating_user_preferences = heating_user_preferences
        self.ev_departure_plans = ev_departure_plans
        self.model_parameters = model_parameters

        self.shutdown_flag = False
        self.cond = threading.Condition()

        app_log_handler = logging.FileHandler("user_app.log")
        app_log_formatter = logging.Formatter("")
        app_log_handler.setFormatter(app_log_formatter)
        self.app_logger = logging.Logger("user_app")
        self.app_logger.addHandler(app_log_handler)

        if self.use_cognit:
            self.init_cognit_runtime()

        self.app_thread = threading.Thread(target=self.app_loop)

    def init_cognit_runtime(self) -> None:
        self.cognit_logger = logging.getLogger("cognit-logger")
        self.cognit_logger.handlers.clear()
        handler = logging.FileHandler("cognit.log")
        formatter = logging.Formatter(
            fmt="[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.cognit_logger.addHandler(handler)

        self.runtime = device_runtime.DeviceRuntime("cognit.yml")
        self.runtime.init(REQS_INIT)

        self.cognit_logger.info("Runtime should be ready now!")

    def set_heating_user_preferences(self, room: str, pref: HeatingPreferences):
        self.heating_user_preferences[room] = pref

    def offload_now(self):
        with self.cond:
            self.offload()
            self.cond.notify_all()

    def set_cycle_length(self, cycle: int):
        with self.cond:
            self.cycle_time = cycle
            self.cond.notify_all()

    def set_speedup(self, speedup: int):
        with self.cond:
            self.speedup = speedup
            self.cond.notify_all()

    def update_algo_input(self, now: float) -> AlgoParams:
        step_timedelta_s = math.floor((now - self.last_algo_run) * self.speedup)
        self.last_algo_run = now
        storage_parameters = self.energy_storage.get_info()
        ev_parameters = self.electric_vehicle.get_info()
        ev_parameters["time_until_charged"] = self.ev_departure_plans.get_time_until_departure()
        room_heating_params_list = []
        for room, value in self.heating_user_preferences.items():
            params = self.room_heating[room].get_info()
            params["preferred_temp"] = value.get_temp()
            room_heating_params_list.append(params)

        energy = self.metrology.get_energy_total()
        energy_drawn_from_grid = energy.active_plus - self.last_active_plus
        energy_returned_to_grid = energy.active_minus - self.last_active_minus
        self.last_active_plus = energy.active_plus
        self.last_active_minus = energy.active_minus

        pv_reg = self.pv.get_info()["energy_produced"]
        energy_pv_produced = pv_reg - self.last_pv_energy
        self.last_pv_energy = pv_reg

        temp_outdoor = self.temp_outside_sensor.get_info()["temperature"]

        charge_level_of_storage = self.energy_storage.get_info()["curr_charge_level"]
        prev_charge_level_of_storage = self.last_storage_charge_level
        self.last_storage_charge_level = charge_level_of_storage

        charge_level_of_ev_battery = self.electric_vehicle.get_info()["curr_charge_level"]
        prev_charge_level_of_ev_battery = self.last_ev_battery_charge_level
        self.last_ev_battery_charge_level = charge_level_of_ev_battery

        heating_status_per_room = {}
        temp_per_room = {}
        for room in room_heating_params_list:
            heating_status_per_room[room["name"]] = room["is_device_switch_on"]
            temp_per_room[room["name"]] = room["curr_temp"]

        algo_input = AlgoParams(
            self.model_parameters,
            step_timedelta_s,
            storage_parameters,
            ev_parameters,
            room_heating_params_list,
            energy_drawn_from_grid / 3.6e6,
            energy_returned_to_grid / 3.6e6,
            energy_pv_produced / 3.6e6,
            temp_outdoor,
            charge_level_of_storage,
            prev_charge_level_of_storage,
            charge_level_of_ev_battery,
            prev_charge_level_of_ev_battery,
            heating_status_per_room,
            temp_per_room,
        )
        return algo_input

    def execute_algo_response(self, algo_res: Any):
        (
            conf_temp_per_room,
            storage_params,
            ev_params,
            next_temp_per_room,
            next_charge_level_of_storage,
            next_charge_level_of_ev_battery,
            energy_from_power_grid,
        ) = algo_res
        self.energy_storage.set_params(storage_params)
        self.electric_vehicle.set_params(ev_params)
        for key, value in self.room_heating.items():
            value.set_params(
                {
                    "optimal_temp": conf_temp_per_room[key],
                }
            )

    def run_algo(self, algo_input: AlgoParams) -> Any:
        ret = None
        if not self.use_cognit:
            ret = self.decision_algo(*astuple(algo_input))
        else:
            try:
                return_code, ret = self.runtime.call(self.decision_algo, *astuple(algo_input))
            except:
                pass
        return ret

    def start(self):
        self.start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.app_thread.start()

    def destroy(self):
        self.shutdown_flag = True
        with self.cond:
            self.cond.notify_all()
        self.app_thread.join()

    def offload(self):
        now = time.clock_gettime(time.CLOCK_MONOTONIC)
        algo_input = self.update_algo_input(now)
        algo_res = self.run_algo(algo_input)

        self.app_logger.info("\n\x1B[2J\x1B[H")
        self.app_logger.info(f"{timedelta(seconds=self.metrology.get_uptime())}")
        self.app_logger.info("\n\tINPUT")
        model_parameters = algo_input.model_parameters
        self.app_logger.info(f"Step duration (s): {algo_input.step_timedelta_s}")
        self.app_logger.info(
            f"Model parameters: \n\t- heat capacity (J/K): {model_parameters['heat_capacity']}, "
            f"\n\t- heating delta temperature (K): {model_parameters['heating_delta_temperature']}, "
            f"\n\t- heating coefficient: {model_parameters['heating_coefficient']}, "
            f"\n\t- heat loss coefficient (W/K): {model_parameters['heat_loss_coefficient']}, "
            f"\n\t- delta charging power (%): {model_parameters['delta_charging_power_perc']}, "
            f"\n\t- storage high SOC (%): {model_parameters['storage_high_charge_level']}"
        )
        storage_parameters = algo_input.storage_parameters
        self.app_logger.info(
            f"Storage parameters: \n\t- max capacity (kWh): {storage_parameters['max_capacity']}, "
            f"\n\t- minimal SOC (%): {storage_parameters['min_charge_level']}, "
            f"\n\t- nominal power (kW): {storage_parameters['nominal_power']}, "
            f"\n\t- efficiency: {storage_parameters['efficiency']}"
        )
        ev_battery_parameters = algo_input.ev_battery_parameters
        time_until_ev_charged = ev_battery_parameters['time_until_charged']
        self.app_logger.info(
            f"EV battery parameters: \n\t- max capacity (kWh): {ev_battery_parameters['max_capacity']}, "
            f"\n\t- is available: {ev_battery_parameters['is_available']}, "
            f"\n\t- departure SOC (%): {ev_battery_parameters['charged_level']}, "
            f"\n\t- time until charged (h): "
            f"{round(time_until_ev_charged / 3600, 2) if time_until_ev_charged > 0 else 'uknown'}, "
            f"\n\t- nominal power (kW): {ev_battery_parameters['nominal_power']}, "
            f"\n\t- efficiency: {ev_battery_parameters['efficiency']}"
        )
        heating_parameters = algo_input.room_heating_params_list[0]
        self.app_logger.info(
            f"Heating parameters: \n\t- current temperature (°C): {round(heating_parameters['curr_temp'], 2)}, "
            f"\n\t- preferred temperature (°C): {heating_parameters['preferred_temp']}, "
            f"\n\t- powers of heating devices (kW): {heating_parameters['powers_of_heating_devices']}, "
            f"\n\t- status of heating devices switches: {heating_parameters['is_device_switch_on']}"
        )
        self.app_logger.info(f"Energy A+ (kWh): {round(algo_input.energy_drawn_from_grid, 2)}")
        self.app_logger.info(f"Energy A- (kWh): {round(algo_input.energy_returned_to_grid, 2)}")
        self.app_logger.info(f"Energy PV produced (kWh): {round(algo_input.energy_pv_produced, 2)}")
        self.app_logger.info(f"Outdoor temperature (°C): {algo_input.temp_outdoor}")
        self.app_logger.info(
            f"Current storage SOC (%): {round(algo_input.charge_level_of_storage, 2)}"
        )
        self.app_logger.info(
            f"Previous storage SOC (%): {round(algo_input.prev_charge_level_of_storage, 2)}"
        )
        self.app_logger.info(
            f"Current EV battery SOC (%): {round(algo_input.charge_level_of_ev_battery, 2)}"
        )
        self.app_logger.info(
            f"Previous EV battery SOC (%): {round(algo_input.prev_charge_level_of_ev_battery, 2)}"
        )

        if algo_res is not None:
            self.execute_algo_response(algo_res)
            self.app_logger.info("\n\tOUTPUT")
            room_name = heating_parameters["name"]
            self.app_logger.info(
                f"Configuration of temperature (°C): {round(algo_res[0][room_name], 2)}"
            )
            self.app_logger.info(f"Configuration of storage: {algo_res[1]}")
            self.app_logger.info(f"Configuration of EV battery: {algo_res[2]}")
            self.app_logger.info(f"Predicted temperature (°C): {round(algo_res[3][room_name], 2)}")
            self.app_logger.info(f"Predicted SOC of storage (%): {round(algo_res[4], 2)}")
            self.app_logger.info(f"Predicted SOC of EV battery (%): {round(algo_res[5], 2)}")
            self.app_logger.info(f"Predicted energy A+ (kWh): {round(algo_res[6], 2)}")
        else:
            self.app_logger.warning(f"Decision algorithm call failed")

    def app_loop(self):
        self.last_algo_run = time.clock_gettime(time.CLOCK_MONOTONIC)
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

                self.offload()
