START_DATE = '2024-09-23'
STOP_DATE = '2024-10-10'

HOUR = 3600
LOOP = 24 * 3600

HEATING_PREFERENCES = [
    (0 * HOUR, 18),  # From 0:00 to 7:00 user temperature setting is 18 °C
    (7 * HOUR, 20),  # From 7:00 to 9:00 user temperature setting is 20 °C
    (9 * HOUR, 18),  # ...
    (17 * HOUR, 21),
    (23 * HOUR, 19),
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
