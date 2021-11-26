from datetime import datetime

today = datetime.today() # month - date - year
now = datetime.now() # hours - minutes - seconds

current_date = str(today.strftime("%b_%d_%Y"))
current_time = str(now.strftime("%H_%M_%S"))

# month_date_year-hour_minute_second
time_info = current_date + "-" + current_time

# directory of Eatron Platooning Repository
base_path = "/home/resul/Eatron/Codes/Company/carla_challenge/ea202101001_platooning_demo/carla_ws" # Local
# base_path = "/cta/eatron/CarlaChallenge/ea202101001_platooning_demo/carla_ws" # WorkStation

best_model_date = "Nov_07_2021-15_32_09"
best_model_name = "epoch_290.pth"

# network output is whether brake or throttle-steer. note that to use 'action_classifier', some comments has to be uncommented
type_of_training = "brake_classifier"

# inference model directory
trained_model_path = base_path + "/model_checkpoint/" + best_model_date + "/" + type_of_training + "/" + best_model_name

agent_mode = {
    0: "inference", # network model controls an ego vehicle agent
    1: "autopilot", # saves data at each 10 steps
    2: "dagger", # applies dagger metric and saves autopilot measurements
    3: "manual" # control vehicle with keyboard if required
    }

# select one three agent modes
selected_mode = agent_mode[3]

if agent_mode[3]:
    # aggregated data path. will be concatenated with training dataset
    save_data_path = base_path + "/dataset/manual/" + time_info + "/"
else:
    save_data_path = base_path + "/dataset/dagger/" + time_info + "/"

# display front image during data-aggregation
debug = True

# threshold values for dagger metric
steer_metric_threshold = 0.4
throttle_metric_threshold = 0.5
brake_metric_threshold = 0.5

# whether to save the measurement data of auto-pilot
save_autopilot_data = True

# only save auto-pilot data for 40 action steps while braking
red_wait_limit = 40

# weather condition will change randomly every 10 steps
weather_change_interval = 10