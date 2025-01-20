import subprocess
import time


COMMAND = ["python3.10", "demo_runner.py", "--offload", "--scenario", "scenario/scenario_autumn.py"]

processes = []


subprocess.call(["mkdir", "-p", "log"])


def spawn(n, offload_cycle=None):
    cmd = COMMAND.copy()
    if offload_cycle is not None:
        cycle = 3600
        speedup = cycle // offload_cycle
        cmd += ["--cycle", f"{cycle}", "--speedup", f"{speedup}"]

    for _ in range(n):
        proc = subprocess.Popen(COMMAND, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(proc)


def killAll():
    for p in processes:
        p.kill()
        p.wait()
    processes.clear()
