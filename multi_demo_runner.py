import argparse
import json
from pathlib import Path
import subprocess
import threading
from datetime import datetime
import os

# from user_app import OffloadFunStatistics, FUNTION_TYPES
# import user_app

subprocess.call(["mkdir", "-p", "log"])  # TODO:

from household_simulator import HouseholdSimulator
from phoenixsystems.sem.time import TimeMachine

MAX_SEM_NUM = 10000

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--live",
    action="store_true",
    help="use live tweaking of simulation variables",
)
parser.add_argument(
    "--scenario_dir",
    default="scenario",
    help="scenario's configuration directory",
)
parser.add_argument(
    "--sem_num",
    help="spawn N instances of SEM (overrides the scenario's configuration)",
)

cmd_args = parser.parse_args()

with open(f"{cmd_args.scenario_dir}/config.json", "r") as f:
    config = json.load(f)

sem_id_list = config["SEM_ID_LIST"]
start_date = datetime.fromisoformat(config["START_DATE"])
speedup = config["SPEEDUP"]

sem_num = config["SEM_NUM"]
if cmd_args.sem_num is not None:
    sem_num = int(cmd_args.sem_num)
if sem_num <= 0 or sem_num > MAX_SEM_NUM:
    raise Exception("Invalid sem_num argument")


time_machine = TimeMachine(
    start_time=start_date,
    speedup=speedup,
    initialize_stopped=True,
)

print(f"Now: {datetime.fromtimestamp(time_machine.get_time_utc())}")


class TrainingController:
    trainings_in_progress = 0
    training_cb_lock = threading.Lock()
    speedup: int

    def __init__(self, speedup: int):
        self.speedup = speedup

    def training_state_changed(self, state: int) -> None:
        with self.training_cb_lock:
            if state == 1:
                if self.trainings_in_progress == 0:
                    time_machine.set_speedup(1)
                self.trainings_in_progress += 1
            if state == 0:
                self.trainings_in_progress -= 1
                if self.trainings_in_progress == 0:
                    time_machine.set_speedup(self.speedup)


training_controller = TrainingController(speedup=speedup)

households: list[HouseholdSimulator] = []
global offload_errors
offload_errors = []

sem_type_idx = 0
loop_num = 0
for i in range(sem_num):
    sem_id = sem_id_list[sem_type_idx]

    device_id = f"{sem_id}_{loop_num:03d}"

    print(f"Initializing Simulation with id: {i} and device_id: {device_id}")

    hsim = HouseholdSimulator(
        sem_id=sem_id,
        device_id=device_id,
        config_dir=Path(cmd_args.scenario_dir),
        time_machine=time_machine,
        live=cmd_args.live,
        training_state_changed_cb=training_controller.training_state_changed,
    )
    households.append(hsim)

    sem_type_idx += 1
    if sem_type_idx == len(sem_id_list):
        sem_type_idx = 0
        loop_num += 1


for hsim in households:
    hsim.start()  # TODO: add graceful finish

print(f"Starting simulation with PID: {os.getpid()}")
time_machine.resume()  # Start the simulation


# def offload_decision_now(id: int):
#     households[id].offload_decision()


# def offload_training_now(id: int):
#     households[id].offload_training()


# def set_decision_cycle(id: int, cycle_sec: int):
#     households[id].set_decision_cycle(cycle_sec)


# def set_training_cycle(id: int, cycle_sec: int):
#     households[id].set_training_cycle(cycle_sec)


# def get_stats():

#     total = {foo: 0 for foo in user_app.FUNTION_TYPES}
#     total_success = {foo: 0 for foo in user_app.FUNTION_TYPES}

#     print("STATS:")
#     for i in range(len(households)):
#         h = households[i]
#         stats: dict[str, OffloadFunStatistics] = h.get_offload_stats()

#         stats_str = f"H{i:03d} : {h.device_id}"

#         for foo in user_app.FUNTION_TYPES:
#             stats_str += f'\t{foo[:2]}: {stats[foo]["success"]} / {stats[foo]["all"]}'
#             total[foo] += stats[foo]["all"]
#             total_success[foo] += stats[foo]["success"]
#         print(stats_str)

#     print("")
#     total_stats_str = f"TOTAL: \t\t"
#     for foo in user_app.FUNTION_TYPES:
#         total_stats_str += f"\t{foo[:2]}: {total_success[foo]} / {total[foo]}"
#     print(total_stats_str)


# def _update_erros() -> list:
#     new_errors = []
#     for i in range(len(households)):
#         h = households[i]
#         errs = h.get_new_errors()
#         for foo in user_app.FUNTION_TYPES:
#             if errs[foo] is None:
#                 continue
#             for e in errs[foo]:
#                 s = f"{e.timestamp} \t H{i:03d} {foo[:2]} \t {e.msg}"
#                 new_errors.append(s)

#     new_errors.sort()
#     global offload_errors
#     offload_errors += new_errors
#     return new_errors


# def get_new_errors():
#     for e in _update_erros():
#         print(e)


# def get_all_errors():
#     _update_erros()
#     for e in offload_errors:
#         print(e)
