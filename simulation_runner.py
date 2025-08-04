import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import phoenixsystems.sem.metersim as metersim
from home_energy_management.device_simulators.electric_vehicle import ElectricVehicle
from home_energy_management.device_simulators.gateway import Gateway
from home_energy_management.device_simulators.heating import Heating, TempSensor
from home_energy_management.device_simulators.photovoltaic import AbstractPV
from home_energy_management.device_simulators.simple_device import SimpleDevice
from home_energy_management.device_simulators.storage import Storage
from sqlalchemy.cyextension.util import Mapping

log_handler = logging.FileHandler(f"log/{os.getpid()}/simulation.log")
formatter = logging.Formatter("")
log_handler.setFormatter(formatter)
logger = logging.Logger("simulation")
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

class SimulationRunner:
    start_date: datetime
    sem: metersim.Metersim
    speedup: int
    scenario_dir: str

    storage: Storage
    pv: AbstractPV
    consumption_device: SimpleDevice
    heating: Heating
    gateway: Gateway
    electric_vehicle_per_id: Mapping[str, ElectricVehicle]
    other_devices: list[metersim.Device]
    temp_outside: TempSensor

    thread: threading.Thread
    shutdown_flag: bool

    def __init__(
            self,
            start_date: datetime,
            pv: AbstractPV,
            storage: Storage,
            consumption_device: SimpleDevice,
            heating: Heating,
            electric_vehicle_per_id: Mapping[str, ElectricVehicle],
            other_devices: list[metersim.Device],
            temp_outside: TempSensor,
            speedup: int,
            scenario_dir: str,
    ):
        self.start_date = start_date
        self.pv = pv
        self.consumption_device = consumption_device
        self.storage = storage
        self.heating = heating
        self.electric_vehicle_per_id = electric_vehicle_per_id
        self.temp_outside = temp_outside
        self.other_devices = other_devices
        self.speedup = speedup
        self.shutdown_flag = False
        self.scenario_dir = scenario_dir

        self.init_sem()
        self.thread = threading.Thread(target=self.simulation_loop)

    def destroy(self):
        self.shutdown_flag = True
        self.thread.join()
        self.sem.destroy_runner()
        self.sem.destroy()

    def set_speedup(self, speedup: int):
        self.speedup = speedup
        self.sem.set_speedup(speedup)

    def init_sem(self):
        self.sem = metersim.Metersim(Path(self.scenario_dir))
        self.sem.create_runner(False)
        self.sem.set_speedup(self.speedup)

        devices = [
            self.consumption_device,
            *self.electric_vehicle_per_id.values(),
            self.temp_outside,
            *self.other_devices,
            self.heating,
        ]
        self.gateway = Gateway(devices, self.storage, self.pv)
        self.sem.add_device(self.gateway)
        self.gateway.init_mgr()

    def start(self):
        self.sem.resume()
        self.thread.start()

    def log(self, now: int):
        vector = self.sem.get_vector()
        energy = self.sem.get_energy_total()

        logger.info("\x1B[2J\x1B[H")
        logger.info(f"{self.start_date + timedelta(seconds=now)}")
        logger.info(
            f"Smart Energy Meter:"
            f"\n\t- Current (A): {round(vector.i[0].real, 2)}"
            f"\n\t- +A (kWh): {round(energy.active_plus / (3600 * 1000), 2)}"
            f"\n\t- -A (kWh): {round(energy.active_minus / (3600 * 1000), 2)}"
        )
        logger.info(
            f"Energy Storage:"
            f"\n\t- Current (A): {round(self.storage.current[0].real, 2)}"
            f"\n\t- SOC (%): {round(self.storage.get_info()['curr_charge_level'], 2)}"
        )
        if len(self.electric_vehicle_per_id) > 0:
            logger.info(f"Electric Vehicle:")
        for ev_id, electric_vehicle in self.electric_vehicle_per_id.items():
            logger.info(
                f"\t- ID: {ev_id}"
                f"\n\t\t- Is available: {electric_vehicle.get_info()['is_available']}"
                f"\n\t\t- Driving power (kW): {round(electric_vehicle.get_info()['driving_power'], 2)}"
                f"\n\t\t- Current (A): {round(electric_vehicle.current[0].real, 2)}"
                f"\n\t\t- SOC (%): {round(electric_vehicle.get_info()['curr_charge_level'], 2)}"
            )
        logger.info(
            f"Photovoltaic:" 
            f"\n\t- Current (A): {round(self.pv.current[0].real, 2)}"
        )
        logger.info(
            f"Heating:"
            f"\n\t- Current temperature (°C): {round(self.heating.get_info()['curr_temp'], 2)}"
            f"\n\t- Optimal temperature (°C): {round(self.heating.get_info()['optimal_temp'], 2)}"
            f"\n\t- Outside temperature (°C): {round(self.temp_outside.get_temp(now), 2)}"
            f"\n\t- Current (A): {round(self.heating.current[0].real, 2)}"
        )
        logger.info(
            f"Consumption of other devices:"
            f"\n\t- Current (A): {round(self.consumption_device.current[0].real, 2)}"
        )

    def simulation_loop(self):
        start = time.time()
        sec = 0

        while not self.shutdown_flag:
            now = self.sem.get_uptime()
            self.heating.update_state(now)
            self.log(now)

            sec += 1
            time.sleep(sec + start - time.time())
