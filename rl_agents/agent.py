from __future__ import print_function

import carla
import cv2

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from planner import RoutePlanner
from rl_agent import RlAgent


class Agent(AutonomousAgent):
    _agent = None
    _route_assigned = False

    def _init(self):
        self.initialized = True

        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        cv2.namedWindow("rgb-front-FOV-60")

    def setup(self, path_to_conf_file):
        self._route_assigned = False
        self._agent = None

        self._sensor_data = {
            'width': 400,
            'height': 300,
            'fov': 60
        }

        self.initialized = False

    def sensors(self):
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                'id': 'rgb_front'},
        ]
        return sensors

    def get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front_image = rgb_front[:, :, :3]
        cv_front_image = rgb_front_image[:, :, ::-1]

        disp_front_image = cv2.UMat(cv_front_image)

        cv2.imshow("rgb-front-FOV-60", disp_front_image)
        cv2.waitKey(1)
        
        print("input_data: ", input_data)

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = RlAgent(hero_actor)

            return control

        if not self._route_assigned:
            if self._global_plan:
                plan = []

                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))

                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True

        else:
            
            control = self._agent.run_step()

        return control
