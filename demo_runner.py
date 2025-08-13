import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

subprocess.call(["mkdir", "-p", "log"])
subprocess.call(["mkdir", "-p", f"log/{os.getpid()}"])


from home_energy_management.baseline_algorithm import make_decision as baseline_decision_function
from home_energy_management.ppo_algorithm import make_decision as ai_decision_function, training_function
from home_energy_management.device_simulators.device_utils import make_current
from home_energy_management.device_simulators.electric_vehicle import (
    ElectricVehicle,
    LiveEVDriving,
    ScheduledEVDriving,
    LiveEVDeparturePlans,
    ScheduledEVDeparturePlans
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

from simulation_runner import SimulationRunner
from user_app import UserApp

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--live",
    action="store_true",
    help="use live tweaking of simulation variables",
)
parser.add_argument(
    "--offload",
    action="store_true",
    help="enable cognit edge nodes for running decision algorithm",
)
parser.add_argument(
    "--start_date",
    help="start date of simulation in format \"YYYY-MM-DD\"",
)
parser.add_argument(
    "--num_simulation_days",
    help="number of days of simulation data",
)
parser.add_argument(
    "--use_baseline_algorithm",
    action="store_true",
    help="disables usage of AI model for decision-making, uses baseline function instead",
)
parser.add_argument(
    "--speedup",
    help="speedup",
)
parser.add_argument(
    "--cycle",
    help="userapp cycle length",
)
parser.add_argument(
    "--num_cycles_retrain",
    help="userapp number of cycles after which to retrain decision model",
)

cmd_args = parser.parse_args()

with open('scenario/config.json', 'r') as f:
    config = json.load(f)

start_date = datetime.fromisoformat(config["START_DATE"])
num_simulation_days = timedelta(days=config["NUM_SIMULATION_DAYS"])
speedup = config["SPEEDUP"]
sem_id = config["SEM_ID"]
reqs_init = config["REQS_INIT"]
besmart_access_parameters = config["BESMART_PARAMETERS"]
s3_parameters = config["S3_PARAMETERS"]


if cmd_args.start_date is not None:
    start_date = datetime.fromisoformat(cmd_args.start_date)

if cmd_args.num_simulation_days is not None:
    num_simulation_days = cmd_args.num_simulation_days

stop_date = start_date + num_simulation_days

if cmd_args.speedup is not None and cmd_args.cycle is not None:
    speedup = int(cmd_args.speedup)
    userapp_cycle = int(cmd_args.cycle)


with open(f'scenario/{sem_id}.json', 'r') as f:
    sem_config = json.load(f)

userapp_cycle = sem_config["USER_APP_CYCLE_LENGTH"]
num_cycles_retrain = sem_config["NUM_CYCLES_RETRAIN"]
initial_state = sem_config["INITIAL_STATE"]
storage_config = sem_config["STORAGE_CONFIG"]
ev_config = sem_config["EV_CONFIG"]
heating_config = sem_config["HEATING_CONFIG"]
model_parameters = sem_config["MODEL_PARAMETERS"]
model_parameters.update(heating_config)
train_parameters = sem_config["TRAIN_PARAMETERS"]
user_preferences = sem_config["USER_PREFERENCES"]
besmart_parameters = sem_config["BESMART_PARAMETERS"]

s3_parameters["model_filename"] = s3_parameters["model_filename"].format(sem_id)
besmart_parameters.update(besmart_access_parameters)


use_ai_algorithm = ~cmd_args.use_baseline_algorithm

if cmd_args.speedup is not None and cmd_args.cycle is not None:
    speedup = int(cmd_args.speedup)
    userapp_cycle = int(cmd_args.cycle)

if cmd_args.num_cycles_retrain is not None:
    num_cycles_retrain = int(cmd_args.num_cycles_retrain)


# Initialize the devices
other_devices = []

if cmd_args.live:
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


print("Initializing Simulation")
simulation = SimulationRunner(
    start_date=start_date,
    scenario_dir="scenario",
    pv=pv,
    storage=storage,
    consumption_device=consumption,
    heating=heating,
    electric_vehicle_per_id=electric_vehicle_per_id,
    other_devices=other_devices,
    temp_outside=temp_outside_sensor,
    speedup=speedup,
)

print("Initializing User Application")
app = UserApp(
    start_date=start_date,
    metrology=simulation.sem,
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
    speedup=speedup,
    cycle=userapp_cycle,
    num_cycles_retrain=num_cycles_retrain,
    use_cognit=cmd_args.offload,
    reqs_init=reqs_init["AI" if use_ai_algorithm else "baseline"],
    heating_user_preferences=heating_preferences,
    ev_departure_plans=ev_departure_plans,
)


def _print_commands(functions: list[tuple[str, str]], live: list[tuple[str, str]]):
    print(80 * "-")
    print("Available commands:\n")
    desc_offset = max([len(x[0]) for x in functions + live]) + 4
    for fun, desc in functions:
        fun_name = "  " + fun + (desc_offset - 2 - len(fun)) * " "
        wrapper = textwrap.TextWrapper(
            width=80,
            initial_indent=fun_name,
            subsequent_indent=desc_offset * " ",
        )
        desc_wrapper = wrapper.wrap(desc)
        for line in desc_wrapper:
            print(line)
        print()
    if cmd_args.live:
        print(80 * "-")
        print("Functions for live tweaking of the parameters:\n")
        for fun, desc in live:
            fun_name = "  " + fun + (desc_offset - 2 - len(fun)) * " "
            wrapper = textwrap.TextWrapper(
                width=80,
                initial_indent=fun_name,
                subsequent_indent=desc_offset * " ",
            )
            desc_wrapper = wrapper.wrap(desc)
            for line in desc_wrapper:
                print(line)
            print()


# Functions to be used in interactive Python
def print_help():
    functions = [
        (
            "print_help()",
            "prints this help message",
        ),
        (
            "set_speedup(speedup: int)",
            "changes the speedup of the simulation (default speedup is 360)",
        ),
        (
            "set_cycle_length(seconds: int)",
            "changes the frequency of running the decision algorithm",
        ),
        (
            "offload()",
            "performs an unscheduled call of decision algorithm",
        ),
        (
            "finish()",
            "finishes the simulation, deletes Serverless Runtime if present",
        ),
    ]

    live_functions = [
        (
            "set_heating_preferences(temp: float)",
            "sets user preferences of heating",
        ),
        (
            "set_pv_state(current: float)",
            "sets PV production (use negative values for production)",
        ),
        (
            "set_consumption(current: float)",
            "sets consumption of uncontrolled devices (use positive values for consumption)",
        ),
        (
            "set_temp_outside(temp: float)",
            "sets temperature outside",
        ),
        (
            "set_ev_driving_power(ev_name: str, driving_power: float)",
            "sets current driving power of EV with id ev_name",
        ),
        (
            "set_ev_departure_time(ev_name: str, ev_departure_time: str)",
            "sets user-planned EV departure time in format %H:%M when EV with id ev_name must be charged",
        ),
    ]

    if cmd_args.live:
        _print_commands(functions=functions, live=live_functions)
    else:
        _print_commands(functions=functions, live=[])
    print(80 * "-")


def offload():
    app.offload_now()


def set_cycle_length(seconds: int):
    app.set_cycle_length(seconds)


def set_heating_preferences(temp: float):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    app.set_heating_user_preferences(LiveHeatingPreferences(temp))


def set_pv_state(current: float):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    if current > 0.0:
        print("Error: PV cannot consume energy")
        return
    pv.set_state(make_current([current, 0, 0]))


def set_consumption(current: float):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    consumption.set_state(make_current([current, 0, 0]))


def set_temp_outside(temp: float):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    temp_outside_sensor.set_temp(temp)


def set_ev_driving_power(ev_name: str, driving_power: float):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    ev_driving_schedule = LiveEVDriving(driving_power)
    ev = simulation.get_ev(ev_name)
    ev.set_getter_of_driving_power(ev_driving_schedule)


def set_ev_departure_time(ev_name: str, ev_departure_time: str):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    ev_departure_plans[ev_name].update_state(ev_departure_time)


def set_speedup(speedup: int):
    if speedup < 1:
        print("Error: speedup should be >= 1")
        return
    simulation.set_speedup(speedup)
    app.set_speedup(speedup)


def finish():
    app.destroy()
    simulation.destroy()
    print("Finished demo")


print("\n\nSTARTING SIMULATION\n\n")
print(f"PID: {os.getpid()}")
print(
    80 * "-",
    "\nConfiguration:\n",
    f"\n  Speedup: {speedup}",
    f"\n  User app cycle length: {userapp_cycle} seconds",
    "\n  Cognit renewable energy: 50%",
)
if cmd_args.live:
    print(
        f"\n  Temperature outside (°C): {initial_state['live_temp_outside']}",
        f"\n  Heating preferences (°C): {initial_state['heating_preferences']}",
        "\n  Consumption current (A): 0",
        "\n  PV current (A): 0",
        f"\n  EV driving power (kW): {initial_state['ev_driving_power']}",
        f"\n  EV departure time planned: {initial_state['ev_departure_time']}",
    )
simulation.start()
app.start()
print_help()
