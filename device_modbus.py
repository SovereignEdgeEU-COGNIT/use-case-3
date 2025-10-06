import asyncio
from pathlib import Path
import json
import logging
from typing import Callable
import threading

from pymodbus.server import StartAsyncSerialServer
from pymodbus.datastore import (
    ModbusSparseDataBlock,
    ModbusServerContext,
    ModbusDeviceContext,
)

# from pymodbus.device import ModbusDeviceIdentification
from pymodbus.client.mixin import ModbusClientMixin
from pymodbus import __version__ as pymodbus_version
from abc import ABC, abstractmethod

import phoenixsystems.sem.metersim as metersim

from home_energy_management.device_simulators.heating import Heating
from home_energy_management.device_simulators.electric_vehicle import ElectricVehicle, ScheduledEVDeparturePlans
from home_energy_management.device_simulators.storage import Storage
from home_energy_management.device_simulators.photovoltaic import AbstractPV
import serial


ADDRESS_SIMULATION_CONTROL = 0x01
ADDRESS_METERSIN = 0x02
ADDRESS_EV_PREF = 0x03
ADDRESS_HEATING_PREF = 0x04
ADDRESS_HOME_MODEL = 0x05
ADDRESS_FIRST_PV = 0x10
ADDRESS_FIRST_EV = 0x20
ADDRESS_FIRST_STORAGE = 0x30
ADDRESS_FIRST_HEATING = 0x40

REGISTER_INFO = 0x01
REGISTER_CONFIG = 0x41

MAX_REG_COUNT = 123  # in a message

log_handler = logging.StreamHandler()
formatter = logging.Formatter("")
log_handler.setFormatter(formatter)
logger = logging.Logger("pymodbus")
logger.addHandler(log_handler)
logger.setLevel(logging.ERROR)


class ModbusCognitSlaveContext(ModbusDeviceContext):
    def __init__(self, *_args, set_callback=None, di=None, co=None, ir=None, hr=None):
        super().__init__(*_args, di=di, co=co, ir=ir, hr=hr)
        self.set_callback = set_callback

    def setValues(self, fc_as_hex, address, values):
        super().setValues(fc_as_hex, address, values)
        if self.set_callback is not None:
            self.set_callback(fc_as_hex, address, values)


class MbSlave(ABC):
    slave_id: int
    context: ModbusCognitSlaveContext

    @abstractmethod
    def on_set_cb(self, fc_as_hex, address, values) -> None:
        pass

    def __init__(self, slave_id):
        super().__init__()
        self.slave_id = slave_id

    @abstractmethod
    def update_regs(self) -> None:
        pass


class MbSimControl(MbSlave):
    user_pref_changed: int
    offload_freq: int  # sec
    training_freq: int  # sec
    predict_now: int = 0
    train_now: int = 0

    def __init__(
        self,
        slave_id,
        pv_cnt,
        ev_cnt,
        storage_cnt,
        heating_cnt,
        speedup,
    ):
        # Defaults
        self.offload_freq = 3600
        self.training_freq = 12 * 3600  # TODO: uint32
        self.user_pref_changed = 1

        super().__init__(slave_id)
        datablock = ModbusSparseDataBlock(
            {
                (REGISTER_INFO + 1): [
                    pv_cnt,
                    ev_cnt,
                    storage_cnt,
                    heating_cnt,
                    speedup,
                ],
                (REGISTER_CONFIG + 1): [0] * 512,  # User preferences
            }
        )

        self.context = ModbusCognitSlaveContext(
            set_callback=self.on_set_cb, di=datablock, co=datablock, hr=datablock, ir=datablock
        )

    def set_user_pref_changed(self):
        self.user_pref_changed = 1  # This is set to 0, by the client, when it obtains the new configuration

    def offload_predict_now(self):
        self.predict_now = 1
        self.set_user_pref_changed()

    def offload_train_now(self):
        self.train_now = 1
        self.set_user_pref_changed()

    def set_offload_freq(self, cycle_len: int):
        self.offload_freq = cycle_len
        self.train_now = 1
        self.set_user_pref_changed()

    def set_training_freq(self, train_cycle_len: int):
        self.training_freq = train_cycle_len
        self.set_user_pref_changed()

    def on_set_cb(self, fc_as_hex, address, values):
        if address == REGISTER_CONFIG and len(values) == 1 and values[0] == 0:
            self.train_now = 0
            self.predict_now = 0
            self.user_pref_changed = 0

    def update_regs(self):
        payload = [self.user_pref_changed]
        payload += ModbusClientMixin.convert_to_registers(
            [self.offload_freq, self.training_freq], data_type=ModbusClientMixin.DATATYPE.UINT32, word_order="big"
        )
        payload += [self.predict_now, self.train_now]
        self.context.setValues(3, REGISTER_CONFIG, payload)


class MbMetersim(MbSlave):
    sem: metersim.Metersim
    training_state_cb: Callable

    def __init__(self, slave_id, sem, training_state_cb=None):
        super().__init__(slave_id)
        self.sem = sem
        self.training_state_cb = training_state_cb

        datablock = ModbusSparseDataBlock(
            {
                REGISTER_INFO + 1: [0] * 12,
                REGISTER_CONFIG + 1: [0] * 12,
            }
        )

        self.context = ModbusCognitSlaveContext(
            set_callback=self.on_set_cb, di=datablock, co=datablock, hr=datablock, ir=datablock
        )

    def on_set_cb(self, fc_as_hex, address, values):
        if self.training_state_cb is None:
            return
        if address != REGISTER_CONFIG or len(values) != 1:
            return
        self.training_state_cb(values[0])

    def update_regs(self):
        energy = self.sem.get_energy_total()
        time = self.sem.get_time_utc()
        uint64_val = [energy.active_plus, energy.active_minus, time]

        payload = ModbusClientMixin.convert_to_registers(
            uint64_val, data_type=ModbusClientMixin.DATATYPE.UINT64, word_order="big"
        )

        self.context.setValues(3, REGISTER_INFO, payload)


class MbPrefStr(MbSlave):
    content: str

    def __init__(self, slave_id):
        super().__init__(slave_id)
        self.content = ""
        datablock = ModbusSparseDataBlock(
            {
                REGISTER_INFO + 1: [0] * 123,
            }
        )

        self.context = ModbusCognitSlaveContext(
            set_callback=self.on_set_cb, di=datablock, co=datablock, hr=datablock, ir=datablock
        )

    def on_set_cb(self, fc_as_hex, address, values):
        pass

    def set_content(self, content: str):
        self.content = content
        if len(self.content) > 243:
            logger.warning("User pref longer than 243. Watch out!")

    def update_regs(self):
        payload = ModbusClientMixin.convert_to_registers(
            self.content, data_type=ModbusClientMixin.DATATYPE.STRING, word_order="big"
        )

        payload.insert(0, len(self.content))

        self.context.setValues(3, REGISTER_INFO, payload)


class MbEnergyStorage(MbSlave):
    storage: Storage

    def __init__(self, slave_id, storage, init_val):
        super().__init__(slave_id)
        self.storage = storage

        float_val = [init_val["InWRte"], init_val["OutWRte"]]

        regs = ModbusClientMixin.convert_to_registers(float_val, data_type=ModbusClientMixin.DATATYPE.FLOAT32)
        regs.append(init_val["StorCtl"])

        datablock = ModbusSparseDataBlock(
            {
                REGISTER_CONFIG + 1: regs,
                REGISTER_INFO + 1: [0] * 14,
            }
        )

        self.context = ModbusCognitSlaveContext(
            set_callback=self.on_set_cb, di=datablock, co=datablock, hr=datablock, ir=datablock
        )

    def on_set_cb(self, fc_as_hex, address, values):
        if address != REGISTER_CONFIG or len(values) != 5:
            return

        values = self.context.getValues(3, REGISTER_CONFIG, 5)
        float_val = ModbusClientMixin.convert_from_registers(
            values[0:4], data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        cfg = {}
        cfg["InWRte"] = float_val[0]
        cfg["OutWRte"] = float_val[1]
        cfg["StorCtl_Mod"] = values[4]

        logger.info(f"\nSetting storage config:\n{cfg}")
        self.storage.set_params(cfg)

    def update_regs(self):
        info = self.storage.get_info()

        float_val = [
            info["max_capacity"],
            info["min_charge_level"],
            info["charging_switch_level"],
            info["curr_charge_level"],
            info["nominal_power"],
            info["efficiency"],
            info["energy_loss"],
        ]

        payload = ModbusClientMixin.convert_to_registers(
            float_val, data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        self.context.setValues(3, REGISTER_INFO, payload)


class MbEV(MbSlave):
    ev: ElectricVehicle
    ev_departure_plans: ScheduledEVDeparturePlans

    def __init__(self, slave_id, ev, ev_departure_plans, init_val):
        super().__init__(slave_id)
        self.ev = ev
        self.ev_departure_plans = ev_departure_plans

        float_val = [init_val["InWRte"], init_val["OutWRte"]]

        regs = ModbusClientMixin.convert_to_registers(float_val, data_type=ModbusClientMixin.DATATYPE.FLOAT32)
        regs.append(init_val["StorCtl"])

        datablock = ModbusSparseDataBlock(
            {
                REGISTER_CONFIG + 1: regs,
                REGISTER_INFO + 1: [0] * 17,
            }
        )

        self.context = ModbusCognitSlaveContext(
            set_callback=self.on_set_cb, di=datablock, co=datablock, hr=datablock, ir=datablock
        )

    def on_set_cb(self, fc_as_hex, address, values):
        if address != REGISTER_CONFIG or len(values) != 5:
            return

        values = self.context.getValues(3, REGISTER_CONFIG, 5)
        float_val = ModbusClientMixin.convert_from_registers(
            values[0:4], data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        cfg = {}
        cfg["InWRte"] = float_val[0]
        cfg["OutWRte"] = float_val[1]
        cfg["StorCtl_Mod"] = values[4]

        logger.info(f"\nSetting EV config:\n{cfg}")

        self.ev.set_params(cfg)

    def update_regs(self):
        info = self.ev.get_info()

        float_val = [
            info["max_capacity"],
            info["min_charge_level"],
            info["charging_switch_level"],
            info["curr_charge_level"],
            info["nominal_power"],
            info["efficiency"],
            info["energy_loss"],
            info["driving_power"],
        ]

        payload = ModbusClientMixin.convert_to_registers(
            float_val, data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        payload.append(int(info["is_available"]))

        uint64_val = [self.ev_departure_plans.get_time_until_departure()]
        payload += ModbusClientMixin.convert_to_registers(
            uint64_val, data_type=ModbusClientMixin.DATATYPE.UINT64, word_order="big"
        )

        self.context.setValues(3, REGISTER_INFO, payload)


class MbHeating(MbSlave):
    room: Heating
    dev_cnt: int

    def __init__(self, slave_id, room, dev_cnt, init_val):
        super().__init__(slave_id)
        self.room = room
        self.dev_cnt = dev_cnt

        float_val = [0.0]  # TODO: get optimal_temp from simulator

        regs = ModbusClientMixin.convert_to_registers(
            float_val, data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        datablock = ModbusSparseDataBlock(
            {
                REGISTER_CONFIG + 1: regs,
                REGISTER_INFO + 1: [0] * (5 + 3 * dev_cnt),
            }
        )

        self.context = ModbusCognitSlaveContext(
            set_callback=self.on_set_cb, di=datablock, co=datablock, hr=datablock, ir=datablock
        )

    def on_set_cb(self, fc_as_hex, address, values):
        if address != REGISTER_CONFIG or len(values) != 2:
            return

        values = self.context.getValues(3, REGISTER_CONFIG, 2)
        float_val = ModbusClientMixin.convert_from_registers(
            values[0:2], data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        cfg = {}
        cfg["optimal_temp"] = float_val

        logger.info(f"\nSetting heating config:\n{cfg}")

        self.room.set_params(cfg)

    def update_regs(self):
        info = self.room.get_info()

        float_val = [
            info["curr_temp"],
            info["optimal_temp"],
        ]

        float_val += info["powers_of_heating_devices"]

        payload = ModbusClientMixin.convert_to_registers(
            float_val, data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        payload.insert(4, self.dev_cnt)

        payload += info["is_device_switch_on"]

        self.context.setValues(3, REGISTER_INFO, payload)


class MbHomeModel(MbSlave):
    def __init__(self, slave_id, values):
        super().__init__(slave_id)

        float_val = [
            values["min_temp_setting"],
            values["max_temp_setting"],
            values["temp_window"],
            values["heating_coefficient"],
            values["heat_loss_coefficient"],
            values["heat_capacity"],
        ]
        payload = ModbusClientMixin.convert_to_registers(
            float_val, data_type=ModbusClientMixin.DATATYPE.FLOAT32, word_order="big"
        )

        datablock = ModbusSparseDataBlock({REGISTER_INFO + 1: payload})

        self.context = ModbusCognitSlaveContext(
            set_callback=self.on_set_cb, di=datablock, co=datablock, hr=datablock, ir=datablock
        )

    def on_set_cb(self, fc_as_hex, address, values):
        pass

    def update_regs(self):
        pass


class ModbusSimulator:
    context: ModbusServerContext
    devices: list[MbSlave]
    thread: threading.Thread
    mbSimControl: MbSimControl

    def __init__(
        self,
        sem: metersim.Metersim,
        storage: list[Storage],
        ev: list[ElectricVehicle],
        ev_departure_plans: list[ScheduledEVDeparturePlans],
        room: list[Heating],
        speedup: int,
        init_user_pref: dict,
        home_model: dict,
        serial_dev: Path,
        training_state_cb=None,
    ):
        self.devices = []

        # Metersim
        mbSem = MbMetersim(
            ADDRESS_METERSIN,
            sem,
            training_state_cb=training_state_cb,
        )
        self.devices.append(mbSem)

        # Storage
        for idx, dev in enumerate(storage):
            init_val = {"InWRte": 1.0, "OutWRte": 1.0, "StorCtl": 0}
            mbStorage = MbEnergyStorage(ADDRESS_FIRST_STORAGE + idx, dev, init_val)
            self.devices.append(mbStorage)

        # EV
        for idx, dev in enumerate(ev):
            init_val = {"InWRte": 1.0, "OutWRte": 1.0, "StorCtl": 0}
            mbEV = MbEV(ADDRESS_FIRST_EV + idx, dev, ev_departure_plans[str(idx)], init_val)
            self.devices.append(mbEV)

        # Heating
        for idx, dev in enumerate(room):
            mbHeating = MbHeating(ADDRESS_FIRST_HEATING + idx, dev, len(dev.heating_devices_power), [])
            self.devices.append(mbHeating)
        mbHm = MbHomeModel(ADDRESS_HOME_MODEL, home_model)
        self.devices.append(mbHm)

        # User preferences
        mbEvPref = MbPrefStr(ADDRESS_EV_PREF)
        mbHeatingPref = MbPrefStr(ADDRESS_HEATING_PREF)
        self.devices += [mbEvPref, mbHeatingPref]

        mbEvPref.set_content(json.dumps(init_user_pref["ev_driving_schedule"]))
        mbHeatingPref.set_content(json.dumps(init_user_pref["pref_temp_schedule"]))

        # Simulation Control
        self.mbSimControl = MbSimControl(
            slave_id=ADDRESS_SIMULATION_CONTROL,
            pv_cnt=0,
            ev_cnt=len(ev),
            storage_cnt=len(storage),
            heating_cnt=len(room),
            speedup=speedup,
        )
        self.mbSimControl.set_offload_freq(3600)
        self.mbSimControl.set_training_freq(3600 * 24)
        self.devices.append(self.mbSimControl)

        slaves = {}
        for dev in self.devices:
            slaves[dev.slave_id] = dev.context
        self.context = ModbusServerContext(devices=slaves, single=False)

        self.serial_dev = serial_dev
        self.thread = threading.Thread(target=self._run_serial)

    async def _updating_task(self):
        while True:
            for dev in self.devices:
                dev.update_regs()
            await asyncio.sleep(1)

    async def _start_serial(self):
        task_updating = asyncio.create_task(self._updating_task())
        await StartAsyncSerialServer(
            context=self.context,
            port=self.serial_dev.as_posix(),
            stopbits=1,
            bytesize=8,
            parity="N",
            baudrate=9600,
        )
        await task_updating

    def _run_serial(self):
        asyncio.run(self._start_serial())

    def offload_predict_now(self):
        self.mbSimControl.offload_predict_now()

    def offload_train_now(self):
        self.mbSimControl.offload_train_now()

    def set_cycle_length(self, cycle: int):
        self.mbSimControl.set_offload_freq(cycle)

    def set_cycle_train_length(self, cycle: int):
        self.mbSimControl.set_training_freq(cycle)

    def start(self):
        logging.basicConfig()
        log = logging.getLogger("pymodbus")
        log.setLevel(logging.ERROR)
        self.thread.start()
