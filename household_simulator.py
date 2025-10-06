import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timedelta
from typing import Callable

from home_energy_management.baseline_algorithm import make_decision as baseline_decision_function
from home_energy_management.ppo_algorithm import make_decision as ai_decision_function, training_function
from home_energy_management.device_simulators.electric_vehicle import (
    ElectricVehicle,
    LiveEVDriving,
    ScheduledEVDriving,
    LiveEVDeparturePlans,
    ScheduledEVDeparturePlans,
)
from home_energy_management.device_simulators.heating import (
    Heating,
    LiveTempSensor,
    LiveHeatingPreferences,
    ScheduledHeatingPreferences,
)
from home_energy_management.device_simulators.photovoltaic import LivePV
from home_energy_management.device_simulators.simple_device import SimpleLiveDevice
from home_energy_management.device_simulators.storage import Storage
from home_energy_management.device_simulators.utils import prepare_device_simulator_from_data

from phoenixsystems.sem.time import TimeMachine

from simulation_runner import SimulationRunner
from user_app import UserApp
from device_modbus import ModbusSimulator


class HouseholdSimulator:
    sem_id: int
    time_machine: TimeMachine | None
    app: UserApp | None = None
    mbsim: ModbusSimulator | None = None

    def __init__(
        self,
        sem_id: int,
        config_dir: Path,
        time_machine: TimeMachine | None = None,
        live: bool = False,
        training_state_changed_cb: Callable | None = None,
    ):
        self.sem_id = sem_id
        self.time_machine = time_machine

        subprocess.call(["mkdir", "-p", "log"])
        subprocess.call(["mkdir", "-p", f"log/{os.getpid()}/{sem_id}"])

        with open(config_dir / "config.json", "r") as f:
            config = json.load(f)

        start_date = datetime.fromisoformat(config["START_DATE"])
        num_simulation_days = timedelta(days=config["NUM_SIMULATION_DAYS"])
        speedup = config["SPEEDUP"]
        reqs_init = config["REQS_INIT"]
        besmart_access_parameters = config["BESMART_PARAMETERS"]
        s3_parameters = config["S3_PARAMETERS"]
        train_parameters = config["TRAIN_PARAMETERS"]

        stop_date = start_date + num_simulation_days

        with open(config_dir / f"{sem_id}.json", "r") as f:
            sem_config = json.load(f)

        userapp_cycle = sem_config["USER_APP_CYCLE_LENGTH"]
        cycle_train = sem_config["TRAIN_CYCLE_LENGTH"]
        initial_state = sem_config["INITIAL_STATE"]
        storage_config = sem_config["STORAGE_CONFIG"]
        ev_config = sem_config["EV_CONFIG"]
        heating_config = sem_config["HEATING_CONFIG"]
        model_parameters = sem_config["MODEL_PARAMETERS"]
        model_parameters.update(heating_config)
        user_preferences = sem_config["USER_PREFERENCES"]
        besmart_parameters = sem_config["BESMART_PARAMETERS"]

        simulation_config = sem_config["SIMULATION_CONFIG"]
        run_local_userapp = simulation_config["local_userapp"]
        modbus_path = Path(simulation_config["modbus_dev"])
        use_cognit = not simulation_config["no_offload"]
        use_ai_algorithm = not simulation_config["use_baseline_algo"]

        s3_parameters["model_filename"] = s3_parameters["model_filename"].format(sem_id)
        besmart_parameters.update(besmart_access_parameters)

        # Initialize the devices
        other_devices = []

        if live:
            temp_outside_sensor = LiveTempSensor(initial_state["live_temp_outside"])
            pv = LivePV()
            consumption = SimpleLiveDevice()
            heating_preferences = LiveHeatingPreferences(initial_state["heating_preferences"])
            ev_driving = {}
            ev_departure_plans = {}
            for ev_id, ev_initial_state in initial_state["ev_state"].items():
                ev_driving[ev_id] = LiveEVDriving(initial_state["ev_driving_power"])
                ev_departure_plans[ev_id] = LiveEVDeparturePlans(initial_state["ev_departure_time"])
            other_devices.append(*ev_departure_plans.values())
        else:
            besmart_parameters["since"] = start_date.timestamp()
            besmart_parameters["till"] = stop_date.timestamp()
            consumption = prepare_device_simulator_from_data(besmart_parameters, "energy_consumption")
            temp_outside_sensor = prepare_device_simulator_from_data(besmart_parameters, "temperature")
            pv = prepare_device_simulator_from_data(besmart_parameters, "pv_generation")
            heating_preferences = ScheduledHeatingPreferences(user_preferences["pref_temp_schedule"])
            ev_driving = {}
            ev_departure_plans = {}
            for ev_id, ev_power_config in user_preferences["ev_driving_schedule"].items():
                ev_driving[ev_id] = ScheduledEVDriving(ev_power_config)
                ev_departure_plans[ev_id] = ScheduledEVDeparturePlans(ev_power_config)
            other_devices.extend([heating_preferences, *ev_driving.values(), *ev_departure_plans.values()])

        storage = Storage(
            max_power=storage_config["max_power"],
            max_capacity=storage_config["max_capacity"],
            min_charge_level=storage_config["min_charge_level"],
            charging_switch_level=storage_config["charging_switch_level"],
            efficiency=storage_config["efficiency"],
            energy_loss=storage_config["energy_loss"],
            current=[0.0, 0.0, 0.0],
            curr_capacity=initial_state["storage_capacity"],
            max_charge_rate=1.0,
            max_discharge_rate=1.0,
            operation_mode=2,
            last_capacity_update=0,
            voltage=[0.0, 0.0, 0.0],
        )

        electric_vehicle_per_id = {}
        for ev_id, ev_config_per_id in ev_config.items():
            ev_initial_state = initial_state["ev_state"][ev_id]
            electric_vehicle_per_id[ev_id] = ElectricVehicle(
                max_power=ev_config_per_id["max_power"],
                max_capacity=ev_config_per_id["max_capacity"],
                min_charge_level=ev_config_per_id["min_charge_level"],
                driving_charge_level=ev_config_per_id["driving_charge_level"],
                charging_switch_level=ev_config_per_id["charging_switch_level"],
                efficiency=ev_config_per_id["efficiency"],
                energy_loss=ev_config_per_id["energy_loss"],
                is_available=ev_initial_state["ev_driving_power"] == 0.0,
                get_driving_power=ev_driving[ev_id].get_driving_power,
                current=[0, 0, 0],
                curr_capacity=ev_initial_state["ev_battery_capacity"],
                max_charge_rate=1.0,
                max_discharge_rate=1.0,
                operation_mode=0,
                last_capacity_update=0,
                voltage=[0, 0, 0],
            )

        heating = Heating(
            heat_capacity=heating_config["heat_capacity"],
            heating_coefficient=heating_config["heating_coefficient"],
            heat_loss_coefficient=heating_config["heat_loss_coefficient"],
            name="room",
            temp_window=heating_config["temp_window"],
            heating_devices_power=heating_config["heating_devices_power"],
            curr_temp=initial_state["curr_room_temp"],
            is_device_switch_on=[False, False],
            optimal_temp=initial_state["heating_preferences"],
            last_temp_update=0,
            current=[0.0, 0.0, 0.0],
            get_temp_outside=temp_outside_sensor.get_temp,
        )

        print(f"Initializing Simulation for sem_id: {sem_id}")
        self.simulation = SimulationRunner(
            start_date=start_date,
            time_machine=time_machine,
            scenario_dir="scenario",
            pv=pv,
            storage=storage,
            consumption_device=consumption,
            heating=heating,
            electric_vehicle_per_id=electric_vehicle_per_id,
            other_devices=other_devices,
            temp_outside=temp_outside_sensor,
            speedup=speedup,
            sem_id=sem_id,
        )

        if run_local_userapp:
            self.app = UserApp(
                sem_id=sem_id,
                start_date=start_date,
                metrology=self.simulation.sem,
                decision_algo=ai_decision_function if use_ai_algorithm else baseline_decision_function,
                model_parameters=model_parameters,
                besmart_parameters=besmart_parameters,
                use_model=use_ai_algorithm,
                training_algo=training_function if use_ai_algorithm else None,
                s3_parameters=s3_parameters if use_ai_algorithm else None,
                train_parameters=train_parameters if use_ai_algorithm else None,
                user_preferences=user_preferences,
                pv=pv,
                electric_vehicle_per_id=electric_vehicle_per_id,
                energy_storage=storage,
                heating=heating,
                temp_outside_sensor=temp_outside_sensor,
                cycle=userapp_cycle,
                cycle_train=cycle_train,
                use_cognit=use_cognit,
                reqs_init=reqs_init["AI" if use_ai_algorithm else "baseline"],
                heating_user_preferences=heating_preferences,
                ev_departure_plans=ev_departure_plans,
                training_state_cb=training_state_changed_cb,
            )
        else:
            self.modbus_path = modbus_path
            self.mbsim = ModbusSimulator(
                sem=self.simulation.sem,
                storage=[storage],
                ev=list(electric_vehicle_per_id.values()),
                ev_departure_plans=ev_departure_plans,
                room=[heating],
                speedup=speedup,
                init_user_pref=user_preferences,
                home_model=model_parameters,
                serial_dev=modbus_path,
                training_state_cb=training_state_changed_cb,
            )
            self.mbsim.mbSimControl.set_offload_freq(userapp_cycle)
            self.mbsim.mbSimControl.set_training_freq(cycle_train)

    def start(self):
        self.simulation.start()

        if self.app is not None:
            self.app.start()

        if self.mbsim is not None:
            self.mbsim.start()

    def offload_decision(self):
        if self.app is not None:
            self.app.offload_predict_now()
        else:
            self.mbsim.offload_predict_now()

    def offload_training(self):
        if self.app is not None:
            self.app.offload_train_now()
        else:
            self.mbsim.offload_train_now()

    def set_decision_cycle(self, cycle: int):
        if self.app is not None:
            self.app.set_cycle_length(cycle)
        else:
            self.mbsim.set_cycle_length(cycle)

    def set_training_cycle(self, cycle: int):
        if self.app is not None:
            self.app.set_cycle_train_length(cycle)
        else:
            self.mbsim.set_cycle_train_length(cycle)
