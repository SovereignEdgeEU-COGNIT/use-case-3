# Cognit Energy Use Case  Basic Demo

## Running the demo

To prepare the environment please follow the steps:

* Prepare Python virtual environment:

```bash
python3 -m venv demo_venv
source demo_venv/bin/activate
pip3 install -r requirements.txt
```

* Inside this directory put the cognit runtime config file `cognit.yml` (with all the necessary authorization data).

* Remember that Cognit SLR uses IPv6, so take care of that if necessary.

In order to run the demo call:

```bash
python3 -i demo_runner.py [--live][--offload][--scenario SCENARIO]
```

Optional arguments:

* `live` – disables the schedules for PV and auto-consumption states, heating preferences and temperature outside. Enables usage of functions that change these parameters live.
* `offload` – enables usage of Cognit Serverless Runtime for offloading the decision algorithm
* `scenario` – enables providing scenario file

NOTE: One should always use exactly one of the `--live` and `--scenario` options.

## Running multiple instances of the simulation

Running multiple instances can be used to simulate the scenario in which a vast number of devices tries to communicate with the COGNIT framework.

In the file `runner.py` the following functions are provided:

* `spawn(n: int, offload_cycle: int)` – spawns `n` instances that offloads a function every `offload_cycle` seconds.
* `status()` – lists all instances PIDs and checks whether some processess terminated
* `killAll()` – kills all the spawned instances

## Running using `docker`

The `DOCKERFILE` is provided in order to simplify the process of the environment setup.

## Logging

For each instance three log files are produced inside `log/{instance_pid}/` directory:

* `simulation.log` – current state of the devices and the simulation environment
* `user_app.log` – input and output of the decision algorithm
* `cognit.log` – Cognit Serverless Runtime device API logs

Additionally in file `log/cognit.log` all the instances register each offload action and whether offloading succedeed.

## Configuration and scenario

The configuration of the simulation, user application and devices can be found in `scenario/config.py`. The most important parameters are:

* `SPEEDUP` (default: 360) – the speedup factor of the simulation
* `USER_APP_CYCLE_LENGTH` (default: 3600 seconds) – time (in "virtual" seconds) between consecutive calls of the decision algorithm

These two parameters can be changed during the simulation (see below).

Files `scenario/config.toml` and `scenario/updates.csv` are SEM Simulator config files (see `../sem-simulator/README.md` for more details).

### Scenarios

There are 4 predefined schedules with consumption approx. 33 kWh/1 day:

* `spring` - temperatures during day between 6 and 18°C, production approx. 60 kWh/1 day
* `summer` - temperatures during day between 17 and 30°C, production approx. 75 kWh/1 day
* `autumn` - temperatures during day between 5 and 15°C, production approx. 50 kWh/1 day
* `winter` - temperatures during day between -5 and 4°C, production approx. 15 kWh/1 day

The proposed values assumes the simulation of a single-story smart home with an area of approx. 150 m2 with an array of PV panels with a power of 15 kWp and an energy storage with a capacity of 24 kWh and a nominal power of 12.8 kW, heated with heating mats with a total power of 16 kW.

For simplicity, in these scenarios all devices are plugged into phase L1.

To understand the structure of scenario files take a look at `scenario/scenario_autumn.py`, where additional comments have been provided.

## Live tweaking of the parameters

We recommend running the demo in interactive Python, as we already suggested above.
In both *live* and *scenario* modes there are the following functions one can use:

* `print_help()` – prints the help message

* `set_speedup(speedup: int)` – changes the speedup of the simulation (by default speedup is 360)

* `set_cycle_length(seconds: int)` – changes the frequency of running the decision algorithm

* `offload()` – performs an unscheduled call of decision algorithm

* `set_slr_config(perc: int)` – updates Serverless Runtime scheduling preferences in terms of green energy usage

* `finish()` – finishes the simulation, deletes Serverless Runtime if present

In *live* mode additionally one can use:

* `set_heating_preferences(temp: float)` – sets user preferences of heating

* `set_pv_state(current: float)` – sets PV production (use negative values for production)

* `set_consumption(current: float)` – sets auto-consumption (use positive values for consumption)

* `set_temp_outside(temp: float)` – sets temperature outside

* `set_ev_driving(driving_power: float)` - sets EV driving power. Setting the power to positive number means that the 
car is driving and its charge level will decrease according to this parameter. However, setting it to 0 is interpreted 
as the car arriving and connecting it to the home charger.

* `set_ev_departure_time(ev_departure_time: str)` - sets user-planned EV departure time in format %H:%M interpreted by
decision algorithm as time when EV must be charged for driving.
