from datetime import timedelta
from pathlib import Path
import threading
import time
import logging
import os

import phoenixsystems.sem.metersim as metersim
from home_energy_management.device_simulators.electric_vehicle import ElectricVehicle
from home_energy_management.device_simulators.heating import RoomHeating, TempSensor
from home_energy_management.device_simulators.storage import Storage
from home_energy_management.device_simulators.gateway import Gateway
from home_energy_management.device_simulators.photovoltaic import AbstractPV
from home_energy_management.device_simulators.simple_device import SimpleDevice

log_handler = logging.FileHandler(f"log/{os.getpid()}/simulation.log")
formatter = logging.Formatter("")
log_handler.setFormatter(formatter)
logger = logging.Logger("simulation")
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

class SimulationRunner:
    sem: metersim.Metersim
    speedup: int
    scenario_dir: str

    storage: Storage
    pv: AbstractPV
    consumption_device: SimpleDevice
    room_heating: dict[str, RoomHeating]
    gateway: Gateway
    electric_vehicle: ElectricVehicle
    other_devices: list[metersim.Device]
    temp_outside: TempSensor

    thread: threading.Thread
    shutdown_flag: bool

    def __init__(
            self,
            pv: AbstractPV,
            storage: Storage,
            consumption_device: SimpleDevice,
            room_heating: dict[str, RoomHeating],
            electric_vehicle: ElectricVehicle,
            other_devices: list[metersim.Device],
            temp_outside: TempSensor,
            speedup: int,
            scenario_dir: str,
    ):
        self.pv = pv
        self.consumption_device = consumption_device
        self.storage = storage
        self.room_heating = room_heating
        self.electric_vehicle = electric_vehicle
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
            self.electric_vehicle,
            self.temp_outside,
            *self.other_devices,
            *self.room_heating.values(),
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
        logger.info(f"{timedelta(seconds=now)}")
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
        logger.info(
            f"Electric Vehicle:"
            f"\n\t- Is available: {self.electric_vehicle.get_info()['is_available']}"
            f"\n\t- Driving power (kW): {round(self.electric_vehicle.get_info()['driving_power'], 2)}"
            f"\n\t- Current (A): {round(self.electric_vehicle.current[0].real, 2)}"
            f"\n\t- SOC (%): {round(self.electric_vehicle.get_info()['curr_charge_level'], 2)}"
        )
        logger.info(
            f"Photovoltaic:" 
            f"\n\t- Current (A): {round(self.pv.current[0].real, 2)}"
        )
        logger.info(
            f"Heating:"
            f"\n\t- Current temperature (°C): {round(self.room_heating['room'].get_info()['curr_temp'], 2)}"
            f"\n\t- Optimal temperature (°C): {round(self.room_heating['room'].get_info()['optimal_temp'], 2)}"
            f"\n\t- Outside temperature (°C): {round(self.temp_outside.get_temp(now), 2)}"
            f"\n\t- Current (A): {round(self.room_heating['room'].current[0].real, 2)}"
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
            self.room_heating["room"].update_state(now)
            self.log(now)

            sec += 1
            time.sleep(sec + start - time.time())
