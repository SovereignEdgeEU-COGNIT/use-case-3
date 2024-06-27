import argparse
import textwrap
import sys

import importlib.util

from home_energy_management.decision_algo import run_one_step
from home_energy_management.device_simulators.device_utils import make_current
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
    MODEL_PARAMETERS,
    STORAGE_CONFIG,
    HEATING_CONFIG,
    INITIAL_STATE,
)
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
    "--scenario",
    help="provide scenario file",
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

    TEMP_OUTSIDE_CONFIG = scenario.TEMP_OUTSIDE_CONFIG
    PV_CONFIG = scenario.PV_CONFIG
    CONSUMPTION_CONFIG = scenario.CONSUMPTION_CONFIG
    HEATING_PREFERENCES = scenario.HEATING_PREFERENCES
    LOOP = scenario.LOOP


# Initialize the devices
other_devices = []

if cmd_args.live:
    temp_outside_sensor = LiveTempSensor(INITIAL_STATE["live_temp_outside"])
    pv = LivePV()
    consumption = SimpleLiveDevice()
    heating_preferences = LiveHeatingPreferences(INITIAL_STATE["heating_preferences"])
else:
    temp_outside_sensor = ScheduledTempSensor(TEMP_OUTSIDE_CONFIG, LOOP)
    pv = ScheduledPV(PV_CONFIG, LOOP)
    consumption = SimpleScheduledDevice(CONSUMPTION_CONFIG, LOOP)
    heating_preferences = ScheduledHeatingPreferences(HEATING_PREFERENCES, LOOP)
    other_devices.append(heating_preferences)

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
    other_devices=other_devices,
    temp_outside=temp_outside_sensor,
    speedup=SPEEDUP,
)


print("Initializing User Application")
app = UserApp(
    metrology=simulation.sem,
    decision_algo=run_one_step,
    model_parameters=MODEL_PARAMETERS,
    pv=pv,
    energy_storage=storage,
    room_heating=room_heating,
    temp_outside_sensor=temp_outside_sensor,
    speedup=SPEEDUP,
    cycle=USER_APP_CYCLE_LENGTH,
    use_cognit=cmd_args.offload,
    heating_user_preferences={
        "room": heating_preferences,
    },
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
    )
simulation.start()
app.start()
print_help()
