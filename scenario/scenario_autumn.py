START_DATE = '2024-09-23'

HOUR = 3600
LOOP = 24 * 3600

# production 50 kWh/1 day
PV_CONFIG = [
    (0 * HOUR, [0, 0, 0]),
    (1 * HOUR, [0, 0, 0]),
    (2 * HOUR, [0, 0, 0]),
    (3 * HOUR, [0, 0, 0]),
    (4 * HOUR, [0, 0, 0]),
    (5 * HOUR, [0, 0, 0]),
    (6 * HOUR, [0, 0, 0]),
    (7 * HOUR, [-1, 0, 0]),  # From 7:00 to 8:00 PV produces current 1 A on phase L1
    (8 * HOUR, [-2, 0, 0]),  # From 8:00 to 9:00 PV produces current 2 A on phase L1
    (9 * HOUR, [-11, 0, 0]),  # ...
    (10 * HOUR, [-47, 0, 0]),
    (11 * HOUR, [-50, 0, 0]),
    (12 * HOUR, [-46, 0, 0]),
    (13 * HOUR, [-27, 0, 0]),
    (14 * HOUR, [-17, 0, 0]),
    (15 * HOUR, [-16, 0, 0]),
    (16 * HOUR, [-4, 0, 0]),
    (17 * HOUR, [-1, 0, 0]),
    (18 * HOUR, [0, 0, 0]),
    (19 * HOUR, [0, 0, 0]),
    (20 * HOUR, [0, 0, 0]),
    (21 * HOUR, [0, 0, 0]),
    (22 * HOUR, [0, 0, 0]),
    (23 * HOUR, [0, 0, 0]),
]

TEMP_OUTSIDE_CONFIG = [
    (0 * HOUR, 7),
    (3 * HOUR, 5),  # From 3:00 to 9:00 temperature outside is 5 °C
    (9 * HOUR, 8),  # From 9:00 to 11:00 temperature outside is 8 °C
    (11 * HOUR, 11),  # ...
    (13 * HOUR, 13),
    (15 * HOUR, 15),
    (18 * HOUR, 13),
    (21 * HOUR, 11),
]

HEATING_PREFERENCES = [
    (0 * HOUR, 18),  # From 0:00 to 7:00 user temperature setting is 18 °C
    (7 * HOUR, 20),  # From 7:00 to 9:00 user temperature setting is 20 °C
    (9 * HOUR, 18),  # ...
    (17 * HOUR, 21),
    (23 * HOUR, 19),
]

# Consumption of other devices (i.e. without heating)
# 33 kWh/1 day
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

# Power of EV driving - positive value implies that the car is driving and energy in its battery is consumed, while 0
# means it is connected to the charger at home. Plans of EV departure are also read from this schedule as next time of
# non-zero power.
EV_POWER_CONFIG = {
    "0": [
        (0 * HOUR, 0),
        (8 * HOUR, 5),
        (15 * HOUR, 0),
        (20 * HOUR, 8),
        (22 * HOUR, 0),
    ],
}
