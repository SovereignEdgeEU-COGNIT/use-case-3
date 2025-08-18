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
python3 -i demo_runner.py [--live][--offload][--start_date DATE][--num_simulation_days NUM_DAYS][--algorithm_version ALG_VER][--speedup SPEEDUP][--cycle CYCLE][--num_cycles_retrain NUM_CYCLES]
```

Optional arguments:

* `live` – disables the schedules for PV production and uncontrolled consumption, temperature outside and user preferences of heating and EV departure time. Enables usage of functions that change these parameters live.
* `offload` – enables usage of Cognit Serverless Runtime for offloading the decision algorithm
* `start_date` – provides start date of simulation in format YYYY-MM-DD
* `num_simulation_days` – provides maximum number of days of simulation data
* `algorithm_version` – sets version of decision-making algorithm, between: "baseline" and "AI"
* `speedup` - sets speedup of simulation time
* `cycle` - sets userapp cycle length
* `num_cycles_retrain` - sets userapp number of cycles after which to retrain decision model

## Running multiple instances of the simulation

Running multiple instances can be used to simulate the scenario in which a vast number of devices tries to communicate with the COGNIT framework.

In the file `runner.py` the following functions are provided:

* `spawn(n: int, offload_cycle: int)` – spawns `n` instances that offloads a function every `offload_cycle` seconds.
* `status()` – lists all instances PIDs and checks whether some processes terminated
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

The general configuration of the demo can be found in `scenario/config.py` and more detailed definition of simulation, user application and devices for specific households with smart energy meter in `{sem_id}.json`. 
The most important parameters are:

* `START_DATE` - start date of the simulation
* `SPEEDUP` (default: 360) – the speedup factor of the simulation
* `SEM_ID` - ID of smart energy meter, pointing to the specification file
* `USER_APP_CYCLE_LENGTH` (default: 3600 seconds) – time (in "virtual" seconds) between consecutive calls of the decision algorithm
* `NUM_CYCLES_RETRAIN` (default: 24) - number of cycles after which training of AI model is called (ignored when using baseline version of algorithm)

Parameters `SPEEDUP` and `USER_APP_CYCLE_LENGTH` can be changed during the simulation (see below).

Files `scenario/config.toml` and `scenario/updates.csv` are SEM Simulator config files (see `../sem-simulator/README.md` for more details).

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
* `set_consumption(current: float)` – sets consumption of uncontrolled devices (use positive values for consumption)
* `set_temp_outside(temp: float)` – sets temperature outside
* `set_ev_driving(ev_name: str, driving_power: float)` - sets driving power of EV with id ev_name. Setting the power to positive number means that the 
car is driving and its charge level will decrease according to this parameter. However, setting it to 0 is interpreted as the car arriving and connecting it to the home charger.
* `set_ev_departure_time(ev_name: str, ev_departure_time: str)` - sets user-planned departure time of EV with id ev_name, in format %H:%M interpreted by decision algorithm as time when EV must be charged for driving.
