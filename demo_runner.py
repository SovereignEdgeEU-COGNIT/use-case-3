import argparse
import importlib.util
import os
import subprocess
import sys
import textwrap
from datetime import datetime


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
    RoomHeating,
    ScheduledTempSensor,
    LiveTempSensor,
    LiveHeatingPreferences,
    ScheduledHeatingPreferences,
)
from home_energy_management.device_simulators.photovoltaic import LivePV, ScheduledPV
from home_energy_management.device_simulators.simple_device import SimpleLiveDevice, SimpleScheduledDevice
from home_energy_management.device_simulators.storage import Storage

from simulation_runner import SimulationRunner
from scenario.config import (
    SPEEDUP,
    USER_APP_CYCLE_LENGTH,
    NUM_CYCLES_RETRAIN,
    ALGORITHM_VERSION,
    REQS_INIT,
    MODEL_PARAMETERS,
    TRAIN_PARAMETERS,
    STORAGE_CONFIG,
    EV_CONFIG,
    HEATING_CONFIG,
    INITIAL_STATE,
    TRAINED_MODEL_PATH,
    PV_PRODUCTION_REAL_PATH,
    PV_PRODUCTION_PRED_PATH,
    UNCONTROLLED_CONSUMPTION_REAL_PATH,
    UNCONTROLLED_CONSUMPTION_PRED_PATH,
    TEMP_OUTSIDE_REAL_PATH,
    TEMP_OUTSIDE_PRED_PATH,
)
from user_app import UserApp, TimeSeries

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
    "--scenario",
    help="provide scenario file",
)
parser.add_argument(
    "--algorithm_version",
    help="version of decision algorithm; available options are: {\"baseline\", \"AI\"}",
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

if not cmd_args.live and not cmd_args.scenario:
    print(
        "\nError when parsing arguments",
        "\nProvide scenario or use live mode",
    )
    parser.print_help()
    sys.exit(1)

if cmd_args.live and cmd_args.scenario:
    print(
        "\nError when parsing arguments",
        "\nEither provide scenario or use live mode",
    )
    parser.print_help()
    sys.exit(1)


# Load the scenario
if cmd_args.scenario is not None:
    scenario_spec = importlib.util.spec_from_file_location("scenario", cmd_args.scenario)
    if scenario_spec is None:
        print("Error when reading scenario!")
        sys.exit(1)

    scenario = importlib.util.module_from_spec(scenario_spec)
    scenario_spec.loader.exec_module(scenario)

    START_DATE = scenario.START_DATE
    TEMP_OUTSIDE_CONFIG = scenario.TEMP_OUTSIDE_CONFIG
    PV_CONFIG = scenario.PV_CONFIG
    CONSUMPTION_CONFIG = scenario.CONSUMPTION_CONFIG
    HEATING_PREFERENCES = scenario.HEATING_PREFERENCES
    EV_POWER_CONFIG = scenario.EV_POWER_CONFIG
    LOOP = scenario.LOOP


if cmd_args.algorithm_version is not None:
    if cmd_args.algorithm_version not in ["baseline", "AI"]:
        print(
            "\nError when parsing arguments",
            "\nAvailable options for algorithm_version are: {\"baseline\", \"AI\"}",
        )
        parser.print_help()
        sys.exit(1)
    algorithm_version = cmd_args.algorithm_version
else:
    algorithm_version = ALGORITHM_VERSION

if cmd_args.speedup is not None and cmd_args.cycle is not None:
    speedup = int(cmd_args.speedup)
    userapp_cycle = int(cmd_args.cycle)
else:
    speedup = SPEEDUP
    userapp_cycle = USER_APP_CYCLE_LENGTH

if cmd_args.num_cycles_retrain is not None:
    num_cycles_retrain = int(cmd_args.num_cycles_retrain)
else:
    num_cycles_retrain = NUM_CYCLES_RETRAIN


# Initialize the devices
other_devices = []

if cmd_args.live:
    start_date = datetime.fromisoformat(INITIAL_STATE["start_date"])
    temp_outside_sensor = LiveTempSensor(INITIAL_STATE["live_temp_outside"])
    pv = LivePV()
    consumption = SimpleLiveDevice()
    heating_preferences = LiveHeatingPreferences(INITIAL_STATE["heating_preferences"])
    ev_driving = LiveEVDriving(INITIAL_STATE["ev_driving_power"])
    ev_departure_plans = LiveEVDeparturePlans(INITIAL_STATE["ev_departure_time"])
    other_devices.append(ev_departure_plans)
else:
    start_date = datetime.fromisoformat(START_DATE)
    temp_outside_sensor = ScheduledTempSensor(TEMP_OUTSIDE_CONFIG, LOOP)
    pv = ScheduledPV(PV_CONFIG, LOOP)
    consumption = SimpleScheduledDevice(CONSUMPTION_CONFIG, LOOP)
    heating_preferences = ScheduledHeatingPreferences(HEATING_PREFERENCES, LOOP)
    ev_driving = ScheduledEVDriving(EV_POWER_CONFIG, LOOP)
    ev_departure_plans = ScheduledEVDeparturePlans(EV_POWER_CONFIG, LOOP)
    other_devices.extend([heating_preferences, ev_driving, ev_departure_plans])

storage = Storage(
    max_power=STORAGE_CONFIG["max_power"],
    max_capacity=STORAGE_CONFIG["max_capacity"],
    min_charge_level=STORAGE_CONFIG["min_charge_level"],
    charging_switch_level=STORAGE_CONFIG["charging_switch_level"],
    efficiency=STORAGE_CONFIG["efficiency"],
    energy_loss=STORAGE_CONFIG["energy_loss"],
    current=[0.0, 0.0, 0.0],
    curr_capacity=INITIAL_STATE["storage_capacity"],
    max_charge_rate=1.0,
    max_discharge_rate=1.0,
    operation_mode=2,
    last_capacity_update=0,
    voltage=[0.0, 0.0, 0.0],
)

electric_vehicle = ElectricVehicle(
    max_power=EV_CONFIG["max_power"],
    max_capacity=EV_CONFIG["max_capacity"],
    min_charge_level=EV_CONFIG["min_charge_level"],
    driving_charge_level=EV_CONFIG["driving_charge_level"],
    charging_switch_level=EV_CONFIG["charging_switch_level"],
    efficiency=EV_CONFIG["efficiency"],
    energy_loss=EV_CONFIG["energy_loss"],
    is_available=INITIAL_STATE["ev_driving_power"] == 0.0,
    get_driving_power=ev_driving.get_driving_power,
    current=[0, 0, 0],
    curr_capacity=INITIAL_STATE["ev_battery_capacity"],
    max_charge_rate=1.0,
    max_discharge_rate=1.0,
    operation_mode=0,
    last_capacity_update=0,
    voltage=[0, 0, 0],
)

room_heating = {
    "room": RoomHeating(
        heat_capacity=HEATING_CONFIG["room"]["heat_capacity"],
        heating_coefficient=HEATING_CONFIG["room"]["heating_coefficient"],
        heating_loss=HEATING_CONFIG["room"]["heating_loss"],
        name="room",
        temp_window=HEATING_CONFIG["room"]["temp_window"],
        heating_devices_power=HEATING_CONFIG["room"]["heating_devices_power"],
        curr_temp=INITIAL_STATE["curr_room_temp"],
        is_device_switch_on=[False, False],
        optimal_temp=INITIAL_STATE["heating_preferences"],
        last_temp_update=0,
        current=[0.0, 0.0, 0.0],
        get_temp_outside=temp_outside_sensor.get_temp,
    ),
}


print("Initializing Simulation")
simulation = SimulationRunner(
    scenario_dir="scenario",
    pv=pv,
    storage=storage,
    consumption_device=consumption,
    room_heating=room_heating,
    electric_vehicle=electric_vehicle,
    other_devices=other_devices,
    temp_outside=temp_outside_sensor,
    speedup=SPEEDUP,
)

print("Initializing User Application")
app = UserApp(
    start_date=start_date,
    metrology=simulation.sem,
    decision_algo=baseline_decision_function if algorithm_version == "baseline" else ai_decision_function,
    model_parameters=MODEL_PARAMETERS,
    use_model=algorithm_version == "AI",
    training_algo=training_function if algorithm_version == "AI" else None,
    model_path=TRAINED_MODEL_PATH if algorithm_version == "AI" else None,
    train_parameters=TRAIN_PARAMETERS if algorithm_version == "AI" else None,
    pv=pv,
    electric_vehicle=electric_vehicle,
    energy_storage=storage,
    room_heating=room_heating,
    temp_outside_sensor=temp_outside_sensor,
    speedup=speedup,
    cycle=userapp_cycle,
    num_cycles_retrain=num_cycles_retrain,
    use_cognit=cmd_args.offload,
    reqs_init=REQS_INIT[algorithm_version],
    heating_user_preferences={
        "room": heating_preferences,
    },
    ev_departure_plans=ev_departure_plans,
    pv_production_series=TimeSeries.from_csv(PV_PRODUCTION_REAL_PATH),
    pv_production_pred_series=TimeSeries.from_csv(PV_PRODUCTION_PRED_PATH),
    consumption_series=TimeSeries.from_csv(UNCONTROLLED_CONSUMPTION_REAL_PATH),
    consumption_pred_series=TimeSeries.from_csv(UNCONTROLLED_CONSUMPTION_PRED_PATH),
    temp_outside_series=TimeSeries.from_csv(TEMP_OUTSIDE_REAL_PATH),
    temp_outside_pred_series=TimeSeries.from_csv(TEMP_OUTSIDE_PRED_PATH),
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
            "set_slr_config(perc: int)",
            "updates Serverless Runtime scheduling preferences in terms of green energy usage",
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
            "sets auto-consumption (use positive values for consumption)",
        ),
        (
            "set_temp_outside(temp: float)",
            "sets temperature outside",
        ),
        (
            "set_ev_driving_power(driving_power: float)",
            "sets EV driving power",
        ),
        (
            "set_ev_departure_time(ev_departure_time: str)",
            "sets user-planned EV departure time in format %H:%M when EV must be charged",
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
    app.set_heating_user_preferences("room", LiveHeatingPreferences(temp))


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


def set_ev_driving_power(driving_power: float):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    ev_driving.set_driving_power(driving_power)


def set_ev_departure_time(ev_departure_time: str):
    if not cmd_args.live:
        print("Error: Live mode disabled")
        return
    ev_departure_plans.update_state(ev_departure_time)


def set_speedup(speedup: int):
    if speedup < 1:
        print("Error: speedup should be >= 1")
        return
    simulation.set_speedup(speedup)
    app.set_speedup(speedup)


def set_slr_config(perc: int):
    if not cmd_args.offload:
        print("Error: Cognit SLR not in use")
        return
    app.update_slr_preferences(perc)


def finish():
    app.destroy()
    simulation.destroy()
    print("Finished demo")


print("\n\nSTARTING SIMULATION\n\n")
print(f"PID: {os.getpid()}")
print(
    80 * "-",
    "\nConfiguration:\n",
    f"\n  Speedup: {SPEEDUP}",
    f"\n  User app cycle length: {USER_APP_CYCLE_LENGTH} seconds",
    "\n  Cognit renewable energy: 50%",
)
if cmd_args.live:
    print(
        f"\n  Temperature outside (°C): {INITIAL_STATE['live_temp_outside']}",
        f"\n  Heating preferences (°C): {INITIAL_STATE['heating_preferences']}",
        "\n  Consumption current (A): 0",
        "\n  PV current (A): 0",
        f"\n  EV driving power (kW): {INITIAL_STATE['ev_driving_power']}",
        "\n  EV departure time planned: 08:00",
    )
simulation.start()
app.start()
print_help()
