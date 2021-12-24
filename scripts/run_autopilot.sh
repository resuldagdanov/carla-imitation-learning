#!/bin/bash

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner

export CHALLENGE_TRACK_CODENAME=SENSORS
export RESUME=True
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1

# TODO: change the following exports
# ------------------------------------------------------------------------------------------------------ #
export BASE_CODE_PATH=~/Research/Codes/Carla/carla-imitation-learning
export TEAM_AGENT=${BASE_CODE_PATH}/imitation_agents/agents/action_agent.py
export TEAM_CONFIG=${BASE_CODE_PATH}/checkpoints/models
export ROUTES=${BASE_CODE_PATH}/data/routes/routes_town04_short.xml
export SCENARIOS=${BASE_CODE_PATH}/data/scenarios/all_towns_traffic_scenarios_autopilot.json
export CHECKPOINT_ENDPOINT=${BASE_CODE_PATH}/results/autopilot_result_town04_short.json
export SAVE_PATH=${BASE_CODE_PATH}/datasets/autopilot
# ------------------------------------------------------------------------------------------------------ #

export PYTHONPATH=$PYTHONPATH:${BASE_CODE_PATH}

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--debug=${DEBUG_CHALLENGE} \
--routes=${ROUTES} \
--scenarios=${SCENARIOS}  \
--repetitions=${REPETITIONS} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--track=${CHALLENGE_TRACK_CODENAME} \
--resume=${RESUME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--record=${RECORD_PATH}
