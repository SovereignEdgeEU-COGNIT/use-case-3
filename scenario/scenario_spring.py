START_DATE = '2024-03-21'

HOUR = 3600
LOOP = 24 * 3600

# production 60 kWh/1 day
PV_CONFIG = [
    (0 * HOUR, [0, 0, 0]),
    (1 * HOUR, [0, 0, 0]),
    (2 * HOUR, [0, 0, 0]),
    (3 * HOUR, [0, 0, 0]),
    (4 * HOUR, [0, 0, 0]),
    (5 * HOUR, [0, 0, 0]),
    (6 * HOUR, [0, 0, 0]),
    (7 * HOUR, [-1, 0, 0]),
    (8 * HOUR, [-3, 0, 0]),
    (9 * HOUR, [-7, 0, 0]),
    (10 * HOUR, [-40, 0, 0]),
    (11 * HOUR, [-50, 0, 0]),
    (12 * HOUR, [-50, 0, 0]),
    (13 * HOUR, [-44, 0, 0]),
    (14 * HOUR, [-35, 0, 0]),
    (15 * HOUR, [-22, 0, 0]),
    (16 * HOUR, [-8, 0, 0]),
    (17 * HOUR, [-2, 0, 0]),
    (18 * HOUR, [0, 0, 0]),
    (19 * HOUR, [0, 0, 0]),
    (20 * HOUR, [0, 0, 0]),
    (21 * HOUR, [0, 0, 0]),
    (22 * HOUR, [0, 0, 0]),
    (23 * HOUR, [0, 0, 0]),
]

TEMP_OUTSIDE_CONFIG = [
    (0 * HOUR, 10),
    (3 * HOUR, 6),
    (6 * HOUR, 8),
    (9 * HOUR, 10),
    (11 * HOUR, 13),
    (13 * HOUR, 16),
    (16 * HOUR, 18),
    (18 * HOUR, 16),
    (20 * HOUR, 13),
    (22 * HOUR, 10),
]

HEATING_PREFERENCES = [
    (0 * HOUR, 18),
    (7 * HOUR, 20),
    (9 * HOUR, 18),
    (17 * HOUR, 21),
    (23 * HOUR, 19),
]

# consumption 33 kWh/1 day
CONSUMPTION_CONFIG = [
    (0 * HOUR, [6, 0, 0]),
    (1 * HOUR, [6, 0, 0]),
    (2 * HOUR, [4, 0, 0]),
    (3 * HOUR, [4, 0, 0]),
    (4 * HOUR, [4, 0, 0]),
    (5 * HOUR, [6, 0, 0]),
    (6 * HOUR, [8, 0, 0]),
    (7 * HOUR, [8, 0, 0]),
    (8 * HOUR, [7, 0, 0]),
    (9 * HOUR, [4, 0, 0]),
    (10 * HOUR, [5, 0, 0]),
    (11 * HOUR, [4, 0, 0]),
    (12 * HOUR, [1, 0, 0]),
    (13 * HOUR, [4, 0, 0]),
    (14 * HOUR, [6, 0, 0]),
    (15 * HOUR, [7, 0, 0]),
    (16 * HOUR, [8, 0, 0]),
    (17 * HOUR, [9, 0, 0]),
    (18 * HOUR, [9, 0, 0]),
    (19 * HOUR, [8, 0, 0]),
    (20 * HOUR, [8, 0, 0]),
    (21 * HOUR, [7, 0, 0]),
    (22 * HOUR, [6, 0, 0]),
    (23 * HOUR, [6, 0, 0]),
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
