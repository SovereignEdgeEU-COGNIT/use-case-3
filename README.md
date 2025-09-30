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

In order to run the demo call:

```bash
python3 multi_demo_runner.py
```

## Logging

For each demoprocess three log files are produced inside `log/{pid}/{sem_id}` directory:

* `simulation.log` – current state of the devices and the simulation environment
* `user_app.log` – input and output of the decision algorithm
* `cognit.log` – Cognit Serverless Runtime device API logs

Additionally in file `log/cognit.log` all the instances register each offload action and whether offloading succedeed.

## Configuration and scenario

The general configuration of the demo can be found in `scenario/config.json` and more detailed definition of simulation, user application and devices for specific households with smart energy meter in `{sem_id}.json`. 
The most important parameters are:

* `START_DATE` - start date of the simulation
* `SPEEDUP` (default: 360) – the speedup factor of the simulation
* `SEM_ID_LIST` - list of IDs of smart energy meters, pointing to the specification files
* `USER_APP_CYCLE_LENGTH` (default: 3600 seconds) – time (in "virtual" seconds) between consecutive calls of the decision algorithm
* `NUM_CYCLES_RETRAIN` (default: 24) - number of cycles after which training of AI model is called (ignored when using baseline version of algorithm)

Parameters `SPEEDUP` and `USER_APP_CYCLE_LENGTH` can be changed during the simulation (see below).

Files `scenario/config.toml` and `scenario/updates.csv` are SEM Simulator config files (see `../sem-simulator/README.md` for more details).
