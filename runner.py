import subprocess
import time


COMMAND = ["python3.10", "demo_runner.py", "--offload", "--scenario", "scenario/scenario_autumn.py"]

processes = []


def spawn(n):
    for _ in range(n):
        proc = subprocess.Popen(COMMAND, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(proc)


def killAll():
    for p in processes:
        p.kill()
        p.wait()
    processes.clear()


def check():
    num_of_proc = len(processes)
    running = 0
    for p in processes:
        ret = p.poll()
        if ret != None:
            print("Process", p.pid, "has terminated with", ret)
            processes.remove(p)
        else:
            running += 1
    print("Running", running, "out of", num_of_proc)


def spawn_and_check(n):
    spawn(n)
    while(1):
        check()
        time.sleep(5)
        