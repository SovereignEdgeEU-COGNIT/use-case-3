import argparse
import json
from pathlib import Path
import subprocess
import threading
from datetime import datetime
import time

subprocess.call(["mkdir", "-p", "log"]) # TODO: 

from household_simulator import HouseholdSimulator
from phoenixsystems.sem.time import TimeMachine

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

cmd_args = parser.parse_args()


with open(f"{cmd_args.scenario_dir}/config.json", "r") as f:
    config = json.load(f)

sem_id_list = config["SEM_ID_LIST"]
start_date = datetime.fromisoformat(config["START_DATE"])
speedup = config["SPEEDUP"]


print(start_date)
time_machine = TimeMachine(
    start_time=start_date,
    speedup=speedup,
    initialize_stopped=False,
)

print(f"Now: {time_machine.get_time_utc()}")

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

households = []
for id in sem_id_list:
    hsim = HouseholdSimulator(
        sem_id=id,
        config_dir=Path(cmd_args.scenario_dir),
        time_machine=time_machine,
        live=cmd_args.live,
        training_state_changed_cb=training_controller.training_state_changed,
    )
    households.append(hsim)

for hsim in households:
    hsim.start() # TODO: add graceful finish


def get_household_by_id(sem_id: int) -> HouseholdSimulator | None:
    for h in households:
        if h.sem_id == sem_id:
            return h
    return None


def offload_decision_now(sem_id: int):
    h = get_household_by_id(sem_id)
    h.offload_decision()

def offload_training_now(sem_id: int):
    h = get_household_by_id(sem_id)
    h.offload_training()

def set_decision_cycle(sem_id:int, cycle_sec: int):
    h = get_household_by_id(sem_id)
    h.set_decision_cycle(cycle_sec)

def set_training_cycle(sem_id:int, cycle_sec: int):
    h = get_household_by_id(sem_id)
    h.set_training_cycle(cycle_sec)