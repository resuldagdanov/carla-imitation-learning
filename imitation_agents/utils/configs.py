import os
from datetime import datetime

today = datetime.today() # month - date - year
now = datetime.now() # hours - minutes - seconds

current_date = str(today.strftime("%b_%d_%Y"))
current_time = str(now.strftime("%H_%M_%S"))

# month_date_year-hour_minute_second
time_info = current_date + "-" + current_time

# main repo directory
base_path = "/home/resul/Research/Codes/Carla/carla-imitation-learning/imitation_agents" # TODO: change if required

# saved model's folder name and model name
best_model_date = "Nov_07_2021-15_32_09"
best_model_name = "epoch_290.pth"

# network output is whether brake or throttle-steer. note that to use 'action_classifier', some comments has to be uncommented
type_of_training = "brake_classifier"

# inference model directory
trained_model_path = base_path + "/checkpoints/" + best_model_date + "/" + type_of_training + "/" + best_model_name

agent_mode = {
    0: "inference", # network model controls an ego vehicle agent
    1: "autopilot", # saves data at each 10 steps
    2: "dagger", # applies dagger metric and saves autopilot measurements
    3: "manual" # control vehicle with keyboard if required
    }

# select one three agent modes
selected_mode = agent_mode[1] # TODO: change if required

# aggregated data path. will be concatenated with training dataset
save_data_path = base_path + "/datasets/" + selected_mode + "/" + time_info + "/"

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

# making sure that the base repo directory exists
if not os.path.exists(base_path):
    print("\n[EXITING !]: ", base_path, " do not exist!")
    os._exit(os.EX_OSFILE)
