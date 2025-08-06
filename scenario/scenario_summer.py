START_DATE = '2024-06-22'
STOP_DATE = '2024-07-10'

HOUR = 3600
LOOP = 24 * 3600

HEATING_PREFERENCES = [
    (0 * HOUR, 18),
    (7 * HOUR, 20),
    (9 * HOUR, 18),
    (17 * HOUR, 21),
    (23 * HOUR, 19),
]

EV_POWER_CONFIG = {
    "0": [
        (0 * HOUR, 0),
        (8 * HOUR, 5),
        (15 * HOUR, 0),
        (20 * HOUR, 8),
        (22 * HOUR, 0),
    ],
}
