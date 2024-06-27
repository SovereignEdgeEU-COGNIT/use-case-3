SPEEDUP = 360  # (s)
USER_APP_CYCLE_LENGTH = 3600  # (s)


STORAGE_CONFIG = {
    "max_power": 12.8,  # (kW)
    "max_capacity": 24.0,  # (kWh)
    "min_charge_level": 20.0,  # (%)
    "charging_switch_level": 60.0,
    "efficiency": 0.98,
    "energy_loss": 0.0,
}


HEATING_CONFIG = {
    "room": {
        "heat_capacity": 3.6e7,
        "heating_coefficient": 0.98,
        "heating_loss": 300.0,
        "temp_window": 0.75,
        "heating_devices_power": [8.0, 8.0],
    }
}


INITIAL_STATE = {
    "live_temp_outside": 15.0,  # °C
    "heating_preferences": 21.0,  # °C
    "storage_capacity": 12.0,  # (kWh)
    "curr_room_temp": 19.0,
}


MODEL_PARAMETERS = {
    "heat_capacity": HEATING_CONFIG["room"]["heat_capacity"],
    "heating_delta_temperature": HEATING_CONFIG["room"]["temp_window"],
    "heating_coefficient": HEATING_CONFIG["room"]["heating_coefficient"],
    "heat_loss_coefficient": HEATING_CONFIG["room"]["heating_loss"],
    "storage_delta_power_perc": 5.0,
    "storage_high_charge_level": 90.0,
}
