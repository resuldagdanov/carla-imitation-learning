# carla-imitation-learning
Imitation Learning Model Training in Carla with DAgger

## Contents
1. [Setup](#setup)
2. [Configuration](#configuration)
2. [Running Carla Server](#running-carla-server)
4. [Run Autopilot](#run-autopilot)
5. [Training](#training)
6. [Evaluation](#evaluation)

## Setup
Clone the repo and build the environment
```Shell
git clone --recursive https://github.com/resuldagdanov/carla-imitation-learning.git
cd carla
conda create -n carla python=3.7
pip3 install -r requirements.txt
conda activate carla
```

Merge all submodules
```Shell
git submodule update --remote --merge
```

Download and setup CARLA 0.9.13
```Shell
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
tar -xf CARLA_0.9.13.tar.gz
tar -xf AdditionalMaps_0.9.13.tar.gz
rm CARLA_0.9.13.tar.gz
rm AdditionalMaps_0.9.13.tar.gz
```

## Configuration
Open bashrc and include CARLA_ROOT, LEADERBOARD_ROOT, and SCENARIO_RUNNER_ROOT
```Shell
export CARLA_ROOT=~/carla
export SCENARIO_RUNNER_ROOT=~/carla-imitation-learning/scenario_runner
export LEADERBOARD_ROOT=~/carla-imitation-learning/leaderboard
```

## Running Carla Server
Open terminal in the folder where Carla 0.9.13 is installed.
```Shell
./CarlaUE4.sh --world-port=2000 --resx=600 --resy=400 --quality-level=Epic -vulkan
```

## Run Autopilot
Configure ```./scripts/run_autopilot.sh``` file !
```Shell
chmod 777 -R *
./scripts/run_autopilot.sh
```

## Training
All training files are in ```./imitation_agents/trainings/``` and ```./imitation_agents/networks/```
NOTE: make sure to check out configurations folder (at ```./imitation_agents/utils/```) before training an imitation learning model.
```Shell
chmod 777 -R *
./scripts/imitation_training.sh
```
Models are saved inside ```./checkpoints/models/``` file.

## Evaluation
Spin up a CARLA server (described above).
```Shell
chmod 777 -R *
./scripts/leaderboard_evaluation.sh
```