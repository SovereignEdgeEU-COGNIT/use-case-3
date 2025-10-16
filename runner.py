import subprocess


class Runner:
    COMMAND = ["python3.10", "multi_demo_runner.py"]
    processes = []

    def __init__(self):
        subprocess.call(["mkdir", "-p", "log"])

    def __del__(self):
        self.kill_all()

    def kill_all(self):
        for p in self.processes:
            p.kill()
            p.wait()
        self.processes.clear()
        
    def kill_num(self, n):
        n = max(n, len(self.processes))
        for _ in range(n):
            p = self.processes.pop()
            p.kill()
            p.wait()

    def spawn(self, n):
        for _ in range(n):
            proc = subprocess.Popen(self.COMMAND, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes.append(proc)
            print(f"Proc {proc.pid} created")


runner = Runner()
