from datetime import datetime

today = datetime.today() # month - date - year
now = datetime.now() # hours - minutes - seconds

current_date = str(today.strftime("%b_%d_%Y"))
current_time = str(now.strftime("%H_%M_%S"))

# month_date_year-hour_minute_second
time_info = current_date + "-" + current_time

# path of main repository
base_path = "/home/resul/Research/Codes/Carla/carla-imitation-learning" # Local
# base_path = "/cta/users/mdal2/ea202101001_platooning_demo/trainings" # WorkStation

# dataset for imitation learning (training)
training_data_path = base_path + "/datasets/autopilot/" # Local
# training_data_path = "/cta/eatron/CarlaDatasets/Autopilot/Extracks/" # WorkStation

# dataset for imitation learning (validation)
validation_data_path = base_path + "/datasets/autopilot/" # Local
# validation_data_path = "/cta/eatron/CarlaDatasets/Autopilot/Extracks/" # WorkStation

# aggregated data path. will be concatenated with training dataset
dagger_data_path = base_path + "/datasets/dagger/" # Local
# dagger_data_path = "/cta/eatron/CarlaDatasets/DAgger/Extracks/" # WorkStation

train_towns = ["Town01_Short"] # Local
# train_towns = [ # WorkStation
#         "Town01_Long", "Town01_Short", \
#         "Town02_Long", "Town02_Short", \
#         "Town03_Long", "Town03_Short", \
#         "Town04_Long", "Town04_Short", \
#         "Town05_Long", "Town05_Short", \
#                        "Town10_Short"]

validation_towns = ["Town01_Short"] # Local
# validation_towns = ["Town03_Long_x"] # WorkStation

# whether to aggregate dagger data to training dataset
use_dagger_data = False

# when dagger data is used, make sure to hane this mode to be True
pretrained = False

# location of all trained and saved models
model_save_path = base_path + "/checkpoints/" + time_info + "/"

# training hyperparameters
batch_size = 32
learning_rate = 1e-2
momentum = 0.9
gamma = 0.1
min_milestone = 10
max_milestone = 20
max_number_of_epochs = 300
loss_print_interval = 32
validate_per_n = 3
save_every_n_epoch = 1

