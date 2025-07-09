SPEEDUP = 360  # (s)
USER_APP_CYCLE_LENGTH = 3600  # (s)
NUM_CYCLES_RETRAIN = 24
ALGORITHM_VERSION = "AI"


REQS_INIT = {
    "baseline":
        {
            "FLAVOUR": "EnergyV2",
            "MIN_ENERGY_RENEWABLE_USAGE": 50,
        },
    "AI":
        {
            "FLAVOUR": "EnergyV2__16GB_1CPU",
            "MIN_ENERGY_RENEWABLE_USAGE": 50,
        },
}


STORAGE_CONFIG = {
    "max_power": 12.8,  # (kW)
    "max_capacity": 24.0,  # (kWh)
    "min_charge_level": 10.0,  # (%)
    "charging_switch_level": 60.0,  # (%)
    "efficiency": 0.85,
    "energy_loss": 0.0,
}


EV_CONFIG = {
    "max_power": 6.9,  # (kW)
    "max_capacity": 40.0,  # (kWh)
    "min_charge_level": 10.0,  # (%)
    "driving_charge_level": 80.0,  # (%)
    "charging_switch_level": 75.0,  # (%)
    "efficiency": 0.85,
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
    "start_date": "2023-03-21",  # YYYY-MM-DD
    "curr_room_temp": 19.0,  # (°C)
    "live_temp_outside": 15.0,  # (°C)
    "heating_preferences": 20.0,  # (°C)
    "storage_capacity": 12.0,  # (kWh)
    "ev_battery_capacity": 20.0,  # (kWh)
    "ev_driving_power": 0.0,  # (kW)
    "ev_departure_time": "08:00",  # HH:mm
}


MODEL_PARAMETERS = {
    "heat_capacity": HEATING_CONFIG["room"]["heat_capacity"],
    "heating_delta_temperature": HEATING_CONFIG["room"]["temp_window"],
    "heating_coefficient": HEATING_CONFIG["room"]["heating_coefficient"],
    "heat_loss_coefficient": HEATING_CONFIG["room"]["heating_loss"],
    "delta_charging_power_perc": 5.0,
    "storage_high_charge_level": 90.0,
    "min_temp_setting": 17.,
    "max_temp_setting": 24.,
}

USER_PREFERENCES = {
    "ev_driving_schedule": {
        "time": ["0:00", "8:00", "15:00", "20:00", "22:00"],
        "driving_power": [0., 5., 0., 8., 0.],
    },
    "pref_temp_schedule": {
        "time": ["0:00", "7:00", "9:00", "17:00", "23:00"],
        "temp": [18., 20, 18., 21., 19.],
    },
}

TRAIN_PARAMETERS = {
    "history_timedelta_days": 90,
    "num_episodes": 8000,
    "critic_lr": 0.001,
    "actor_lr": 0.001,
    "gamma": 0.2,  # Discount factor for future rewards
    "lambda_": 0.95,
    "num_epochs": 10,
    "eps_clip": 0.2,
    "min_action_std": 0.1,
    "action_std_decay_freq": 1 / 50,
    "action_std_decay_rate": 0.01,
    "update_epoch": 10,
    "action_std_init": 0.6,
    "batch_size": 64,
    "energy_reward_coeff": 0.3,
    "temp_reward_coeff": 2.0,
    "storage_reward_coeff": 0.8,
    "ev_reward_coeff": 0.8,
}


S3_PARAMETERS = {
    "endpoint_url": "https://s3.sovereignedge.eu/",
    "bucket_name": "uc3-test-bucket",
    "model_filename": "files/onnx_model_from_cognit.onnx",
    "access_key_id": "eXuxY2Gt4bI8PTScQ9gz",
    "secret_access_key": "RtomdwwpoN7tkQe6ZZSPZTGScvQ0GtEwVhObreo4",
}

BESMART_PARAMETERS = {
    "workspace_key": "wubbalubbadubdub",
    "login": "cognit_demo",
    "password": "CognitDemo2025!",
    "pv_generation": {
        "cid": 68,
        "mid": 84,
        "moid": 70,
    },
    "energy_consumption": {
        "cid": 68,
        "mid": 83,
        "moid": 32,
    },
    "temperature_moid": 139,
}