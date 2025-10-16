import argparse
import json
from pathlib import Path
import subprocess
import threading
from datetime import datetime
<<<<<<< HEAD

=======
import time
import pprint

from user_app import OffloadFunStatistics

>>>>>>> ef9b522 (workin)
subprocess.call(["mkdir", "-p", "log"])  # TODO:

from household_simulator import HouseholdSimulator
from phoenixsystems.sem.time import TimeMachine

MAX_MULTIPLY = 100

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
    help="run userapp locally",
)
parser.add_argument(
    "--multiply",
    default=1,
    help="spawn N instances of each SEM from SEM_ID_LIST",
)

cmd_args = parser.parse_args()

with open(f"{cmd_args.scenario_dir}/config.json", "r") as f:
    config = json.load(f)

sem_id_list = config["SEM_ID_LIST"]
start_date = datetime.fromisoformat(config["START_DATE"])
speedup = config["SPEEDUP"]

n_multiply = int(cmd_args.multiply)
if n_multiply <= 0 or n_multiply > MAX_MULTIPLY:
    raise Exception("Invalid multiply argument")


print(start_date)
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

j = 0
for i in range(n_multiply):
    for id in sem_id_list:
        subprocess.call(["mkdir", "-p", f"log/{id}"])  # TODO: proper logging
        print(f"Initializing Simulation {j} for SEM {id}")
        hsim = HouseholdSimulator(
            sem_id=id,
            config_dir=Path(cmd_args.scenario_dir),
            time_machine=time_machine,
            live=cmd_args.live,
            training_state_changed_cb=training_controller.training_state_changed,
        )
        households.append(hsim)
        j += 1

for hsim in households:
    hsim.start()  # TODO: add graceful finish

print("Starting simulation")
time_machine.resume()  # Start the simulation


def offload_decision_now(id: int):
    households[id].offload_decision()


def offload_training_now(id: int):
    households[id].offload_training()


def set_decision_cycle(id: int, cycle_sec: int):
    households[id].set_decision_cycle(cycle_sec)


def set_training_cycle(id: int, cycle_sec: int):
    households[id].set_training_cycle(cycle_sec)


def get_stats():
    ret = []

    total_train = 0
    total_train_success = 0
    total_decision = 0
    total_decision_success = 0

    print("STATS:")
    for i in range(len(households)):
        h = households[i]
        stats: dict[str, OffloadFunStatistics] = h.get_offload_stats()
        print(
            f'H{i} \t {h.sem_id} \t train: {stats["train"]["success"]} / {stats["train"]["all"]}, \t decision: {stats["decision"]["success"]} / {stats["decision"]["all"]}'
        )
        total_train += stats["train"]["all"]
        total_train_success += stats["train"]["success"]
        total_decision += stats["decision"]["all"]
        total_decision_success += stats["decision"]["success"]
    print("")
    print(f"TOTAL: \t\t train: {total_train_success} / {total_train} \t decision: {total_decision_success} / {total_decision}")


def get_new_errors():
    new_errors = []
    for i in range(len(households)):
        h = households[i]
        errs = h.get_new_errors()
        if errs["train"] is not None:
            for e in errs["train"]:
                s = f"{e.timestamp} \t H{i} T \t {e.msg}"
                new_errors.append(s)
        if errs["decision"] is not None:
            for e in errs["decision"]:
                s = f"{e.timestamp} \t H{i} D \t {e.msg}"
                new_errors.append(s)

    new_errors.sort()
    global offload_errors
    offload_errors += new_errors
    for e in new_errors:
        print(e)


def get_all_errors():
    for e in offload_errors:
        print(e)
