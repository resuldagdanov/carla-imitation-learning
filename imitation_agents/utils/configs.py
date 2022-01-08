import os
from datetime import datetime

today = datetime.today() # month - date - year
now = datetime.now() # hours - minutes - seconds

current_date = str(today.strftime("%b_%d_%Y"))
current_time = str(now.strftime("%H_%M_%S"))

# month_date_year-hour_minute_second
time_info = current_date + "-" + current_time

# main repo directory
base_path = os.environ.get('BASE_CODE_PATH', None)

# saved model's folder name and model name
best_model_date = "offset_model/Jan_05_2022-11_31_15"
best_model_name = "epoch_45.pth"

# inference model directory
trained_model_path = base_path + "/checkpoints/models/" + best_model_date + "/" + best_model_name

agent_mode = {
    0: "inference", # network model controls an ego vehicle agent
    1: "autopilot", # saves data at each 10 steps
    2: "dagger", # applies dagger metric and saves autopilot measurements
    3: "manual" # control vehicle with keyboard if required
    }

# select one three agent modes
selected_mode = agent_mode[0] # TODO: change if required

# aggregated data path. will be concatenated with training dataset
save_data_path = base_path + "/datasets/" + selected_mode + "/" + time_info + "/"

# display front image during data-aggregation
debug = True

# threshold values for dagger metric
steer_metric_threshold = 0.4
throttle_metric_threshold = 0.5
brake_metric_threshold = 0.5

# whether to save the measurement data of auto-pilot
save_autopilot_data = False

# manually change autopilot action; works only when manual agent type is active
manual_autopilot = False

# only save auto-pilot data for 40 action steps while braking
red_wait_limit = 40

# weather condition will change randomly every 10 steps
weather_change_interval = 1000

# making sure that the base repo directory exists
if not os.path.exists(base_path):
    print("\n[EXITING !]: ", base_path, " do not exist!")
    os._exit(os.EX_OSFILE)

# make required saving directories
if save_autopilot_data:
    if not os.path.exists(base_path + "/datasets/"):
        os.makedirs(base_path + "/datasets/")
    if not os.path.exists(base_path + "/datasets/inference/"):
        os.makedirs(base_path + "/datasets/inference/")
    if not os.path.exists(base_path + "/datasets/autopilot/"):
        os.makedirs(base_path + "/datasets/autopilot/")
    if not os.path.exists(base_path + "/datasets/dagger/"):
        os.makedirs(base_path + "/datasets/dagger/")
    if not os.path.exists(base_path + "/datasets/manual/"):
        os.makedirs(base_path + "/datasets/manual/")