import glob
import sys
import os
import importlib
import inspect
import carla
import pkg_resources

from distutils.version import LooseVersion
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager


SCENARIO_NAME = 'ControlLoss_1'
CONFIG_FILE_NAME = ''
ADDITIONAL_SCENARIO = ''
HOST = '127.0.0.1'
PORT = 2000

# constants
trafficManagerPort = 8000
trafficManagerSeed = 0
frame_rate = 20 # hz
randomize = True
debug = True
client_timeout = 10.0 # sec


def get_scenario_class_or_fail(scenario):
    # path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument
    scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
    scenarios_list.append(ADDITIONAL_SCENARIO)

    for scenario_file in scenarios_list:

        # get their module
        module_name = os.path.basename(scenario_file).split('.')[0]
        sys.path.insert(0, os.path.dirname(scenario_file))
        scenario_module = importlib.import_module(module_name)

        # and their members of type class
        for member in inspect.getmembers(scenario_module, inspect.isclass):
            if scenario in member:
                return member[1]

        # remove unused Python paths
        sys.path.pop(0)

    print("Scenario '{}' not supported ... Exiting".format(scenario))
    sys.exit(-1)


def prepare_ego_vehicles(world, ego_vehicles):
    ego_vehicle_missing = True

    while ego_vehicle_missing:
        all_ego_vehicles = []
        ego_vehicle_missing = False

        for ego_vehicle in ego_vehicles:
            ego_vehicle_found = False

            carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')

            for carla_vehicle in carla_vehicles:
                if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                    ego_vehicle_found = True
                    all_ego_vehicles.append(carla_vehicle)
                    break

            if not ego_vehicle_found:
                ego_vehicle_missing = True
                break
    
    for i, _ in enumerate(all_ego_vehicles):
        all_ego_vehicles[i].set_transform(ego_vehicles[i].transform)
        CarlaDataProvider.register_actor(all_ego_vehicles[i])

    # TODO:
    # sync state
    if CarlaDataProvider.is_sync_mode():
        world.tick()
    else:
        world.wait_for_tick()

    return all_ego_vehicles


def load_and_wait_for_world(town):
    
    world = carla_client.load_world(town)
    world = carla_client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / frame_rate

    world.apply_settings(settings)

    CarlaDataProvider.set_client(carla_client)
    CarlaDataProvider.set_world(world)

    # TODO:
    # wait for the world to be ready
    if CarlaDataProvider.is_sync_mode():
        world.tick()
    else:
        world.wait_for_tick()

    return world


def load_and_run_scenario(config):

    world = load_and_wait_for_world(config.town)
    
    if not world:
        return False

    CarlaDataProvider.set_traffic_manager_port(int(trafficManagerPort))

    tm = carla_client.get_trafficmanager(int(trafficManagerPort))
    tm.set_random_device_seed(int(trafficManagerSeed))
    tm.set_synchronous_mode(True)

    all_ego_vehicles = prepare_ego_vehicles(world, config.ego_vehicles)

    scenario_class = get_scenario_class_or_fail(config.type)
    scenario = scenario_class(world, all_ego_vehicles, config, randomize, debug)

    manager.load_scenario(scenario=scenario, agent=None)

    # TODO: thread this part
    manager.run_scenario()

    return True


if __name__ == "__main__":

    carla_client = carla.Client(HOST, PORT)
    carla_client.set_timeout(client_timeout)

    dist = pkg_resources.get_distribution("carla")
    if LooseVersion(dist.version) < LooseVersion('0.9.12'):
        raise ImportError("CARLA version 0.9.12 or newer required. CARLA version found: {}".format(dist))

    # create the scenario manager
    manager = ScenarioManager(debug_mode=debug, sync_mode=True, timeout=client_timeout)

    print("scenario manager: ", manager)

    scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(scenario_name=SCENARIO_NAME, config_file_name=CONFIG_FILE_NAME)

    print("scenario_configurations: ", scenario_configurations, len(scenario_configurations))
    print("town: ", scenario_configurations[0].town, "ego_vehicles: ", scenario_configurations[0].ego_vehicles)

    for scenario_config in scenario_configurations:
        result = load_and_run_scenario(scenario_config)
        