import os
import json
import cv2
import random
import carla
import numpy as np
import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_a, K_d, K_s, K_w, K_r, K_t, K_c, K_v
from imitation_agents.base_codes.map_agent import MapAgent
from imitation_agents.base_codes.pid_controller import PIDController
from imitation_agents.utils import base_utils, configs
from imitation_agents.networks.action_model import ActionModel


def get_entry_point():
    return 'ActionAgent'


class ActionAgent(MapAgent):

    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

    def eatron_setup(self):
        self.dataset_save_path = configs.save_data_path
        self.run_type = configs.selected_mode

        self.red_wait_counter = 0
        self.save_counter = 0

        self.debug = configs.debug
        self.save_autopilot = configs.save_autopilot_data
        self.red_wait_limit = configs.red_wait_limit

    def _init(self):
        super()._init()

        self.manual_autopilot = True # manually change autopilot action

        self.dagger_counter = 0
        self.target_vehicle_speed = 0

        # auto-pilot initializer
        self.init_auto_pilot()

        # init eatrun setup and variables
        self.eatron_setup()

        if self.debug is True:
            cv2.namedWindow("rgb-front-FOV-60")
            cv2.namedWindow("rgb-rear-FOV-100")
            cv2.namedWindow("rgb-left-FOV-100")
            cv2.namedWindow("rgb-right-FOV-100")

        # initialize pygame screen
        if self.run_type is "manual":
            pygame.init()
            self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)

            # initially do not save data, save only when T key is pressed
            self.save_autopilot = False
            self.reverse_active = False

        # init agent
        if self.run_type is "dagger" or self.run_type is "inference":
            self.agent = ActionModel()

        if self.run_type is "autopilot" or self.run_type is "dagger" or self.run_type is "manual":
            self.init_dataset(output_dir=self.dataset_save_path)

    def init_auto_pilot(self):
        self._target_stop_sign = None # the stop sign affecting the ego vehicle
        self._stop_completed = False # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign

        self.weather_counter = 0
        self.weather_change_interval = configs.weather_change_interval
        self.weather_id = None

        # PID controllers of auto_pilot
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def init_dataset(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.subfolder_paths = []
        self.data_count = 0

        subfolders = [
            # "rgb_front_100",
            "rgb_front_60",
            "rgb_rear_100",
            # "rgb_rear_60",
            "rgb_left_100",
            # "rgb_left_60",
            "rgb_right_100",
            # "rgb_right_60",
            "measurements"]

        for subfolder in subfolders:
            self.subfolder_paths.append(os.path.join(output_dir, subfolder))
            if not os.path.exists(self.subfolder_paths[-1]):
                os.makedirs(self.subfolder_paths[-1])

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        # change weather for visual diversity
        self._change_weather()

        data = self.tick(input_data)
        gps = self._get_position(data)
        speed = data['speed']

        # fix for theta=nan in some measurements
        if np.isnan(data['compass']):
            ego_theta = 0.0
        else:
            ego_theta = data['compass']

        rgb_front_image = data['rgb_front_60'][:, :, :3]
        cv_front_image = rgb_front_image[:, :, ::-1]

        rgb_rear_image = data['rgb_rear_100'][:, :, :3]
        cv_rear_image = rgb_rear_image[:, :, ::-1]

        rgb_left_image = data['rgb_left_100'][:, :, :3]
        cv_left_image = rgb_left_image[:, :, ::-1]

        rgb_right_image = data['rgb_right_100'][:, :, :3]
        cv_right_image = rgb_right_image[:, :, ::-1]

        image_list = [cv_front_image, cv_rear_image, cv_left_image, cv_right_image]

        if self.debug is True:
            disp_front_image = cv2.UMat(cv_front_image)
            disp_rear_image = cv2.UMat(cv_rear_image)
            disp_left_image = cv2.UMat(cv_left_image)
            disp_right_image = cv2.UMat(cv_right_image)

        # get near and far waypoints from route planners
        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        # get auto-pilot actions
        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data)

        expert_control = carla.VehicleControl()
        expert_control.steer = steer + 1e-2 * np.random.randn()
        expert_control.throttle = throttle
        expert_control.brake = float(brake)

        if self.run_type is "autopilot" or self.run_type is "dagger" or self.run_type is "inference":
            measurement_data = {
                'x': gps[0],
                'y': gps[1],

                'speed': speed,
                'theta': ego_theta,

                'x_command': far_node[0],
                'y_command': far_node[1],
                'far_command': far_command.value,

                'near_node_x': near_node[0],
                'near_node_y': near_node[1],
                'near_command': near_command.value,

                'steer': expert_control.steer,
                'throttle': expert_control.throttle,
                'brake': expert_control.brake,

                'weather_id': self.weather_id,

                'should_slow': self.should_slow,
                'should_brake': self.should_brake,

                'angle': self.angle,
                'angle_unnorm': self.angle_unnorm,
                'angle_far_unnorm': self.angle_far_unnorm,

                'is_vehicle_present': self.is_vehicle_present,
                'is_pedestrian_present': self.is_pedestrian_present,
                'is_red_light_present': self.is_red_light_present,
                'is_stop_sign_present': self.is_stop_sign_present
                }

        # network actions
        self.current_control = carla.VehicleControl()

        # if dagger or test is running inference through network
        if self.run_type is "dagger" or self.run_type is "inference":

            ego_x = measurement_data['x']
            ego_y = measurement_data['y']
            
            # get far node waypoints
            x_command = measurement_data['x_command']
            y_command = measurement_data['y_command']

            # rotation matrix
            R = np.array([
                [np.cos(np.pi/2 + ego_theta), -np.sin(np.pi/2 + ego_theta)],
                [np.sin(np.pi/2 + ego_theta),  np.cos(np.pi/2 + ego_theta)]
                ])

            # convert to local waypoint commands relative to ego agent
            local_command_point = np.array([x_command - ego_x, y_command - ego_y])
            local_command_point = R.T.dot(local_command_point)

            # concatenate vehicle velocity with local far waypoint commands
            fused_inputs = np.zeros(3, dtype=np.float32)
            fused_inputs[0] = measurement_data['speed']
            fused_inputs[1] = local_command_point[0]
            fused_inputs[2] = local_command_point[1]

            # get brake action from network
            dnn_agent_control = self.agent.inference(cv_front_image, fused_inputs)

            self.current_control.throttle = float(expert_control.throttle)
            self.current_control.steer = float(expert_control.steer)
            self.current_control.brake = float(dnn_agent_control)

            # make sure that throttle is model active while braking
            if self.current_control.brake == 1.0:
                self.current_control.throttle = 0.0

        # only auto pilot is used
        if self.run_type is "autopilot":

            print("Expert Actions:", round(expert_control.throttle, 2), round(expert_control.steer, 2), round(expert_control.brake, 2))
            
            # save dataset and return expert control
            if self.save_autopilot is True:
                self.check_and_save(image_list, measurement_data)

            applied_control = expert_control
            
        # use pre-trained imitation learning agent and compare with auto-pilot
        elif self.run_type is "dagger":

            # inference or dagger mode calculate the metric
            dagger_metric = self.check_dagger_brake_metric(expert_control.brake, self.current_control.brake)

            print("Expert Actions:", round(expert_control.throttle, 2), round(expert_control.steer, 2), round(expert_control.brake, 2), \
                 " Agent Actions:", round(self.current_control.throttle, 2), round(self.current_control.steer, 2), round(self.current_control.brake, 2), " DAgger:", dagger_metric)

            if dagger_metric:
                if self.debug:
                    disp_front_image = cv2.putText(disp_front_image, "Auto-Pilot", (0, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))

                # make sure that throttle is uneffective while braking
                if expert_control.brake:
                    expert_control.steer *= 0.5
                    expert_control.throttle = 0.0

                if self.save_autopilot is True:
                    self.check_and_save(image_list, measurement_data)

                applied_control = expert_control

            else:
                if self.debug:
                    disp_front_image = cv2.putText(disp_front_image, "Agent", (0, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))

                applied_control = self.current_control

        elif self.run_type is "inference":
            print("Agent Actions:", round(self.current_control.throttle, 2), round(self.current_control.steer, 2), round(self.current_control.brake, 2))

            applied_control = self.current_control
        
        elif self.run_type is "manual":

            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if self.manual_autopilot:
                self.current_control.throttle = expert_control.throttle
                self.current_control.steer = expert_control.steer
            else:
                self.current_control.throttle = 0
                self.current_control.steer = 0
            
            # brake should be active when there is no throttle
            self.current_control.brake = 1.0

            # save data only when T key is pressed
            if keys[K_t]:
                self.save_autopilot = not self.save_autopilot

            # activate or deactivate reverse gear
            if keys[K_v]:
                self.reverse_active = not self.reverse_active
            
            if keys[K_w]:
                self.current_control.throttle = 0.75
                self.current_control.brake = 0.0

            elif keys[K_s]:
                # make sure that throttle is not active while braking
                self.current_control.brake = 1.0
                self.current_control.throttle = 0.0

            elif keys[K_a]:
                self.current_control.steer = -1

            elif keys[K_d]:
                self.current_control.steer = +1

            # v: reverse or not reverse
            if self.reverse_active:
                self.current_control.reverse = True
            else:
                self.current_control.reverse = False

            measurement_data = {
                'x': gps[0],
                'y': gps[1],

                'speed': speed,
                'target_speed': target_speed,
                'theta': ego_theta,

                'x_command': far_node[0],
                'y_command': far_node[1],
                'far_command': far_command.value,

                'near_node_x': near_node[0],
                'near_node_y': near_node[1],
                'near_command': near_command.value,

                'steer': self.current_control.steer,
                'throttle': self.current_control.throttle,
                'brake': self.current_control.brake,

                'should_slow': self.should_slow,

                'angle': self.angle,
                'angle_unnorm': self.angle_unnorm,
                'angle_far_unnorm': self.angle_far_unnorm,

                'weather_id': self.weather_id
                }

            applied_control = self.current_control

            if self.debug:
                if self.reverse_active:
                    disp_right_image = cv2.putText(disp_right_image, "R-Gear", (0, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
                else:
                    disp_right_image = cv2.putText(disp_right_image, "D-Gear", (0, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))

            if self.save_autopilot is True:
                self.dataset_save(image_list, measurement_data)

                if self.debug:
                    disp_front_image = cv2.putText(disp_front_image, "Being Saved", (0, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
            else:
                if self.debug:
                    disp_front_image = cv2.putText(disp_front_image, "Not Saved", (0, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))

        else:
            print("Error in Agent Mode ! Should be one of the following: ", configs.agent_mode)

        if self.debug is True:
            cv2.imshow("rgb-front-FOV-60", disp_front_image)
            cv2.imshow("rgb-rear-FOV-100", disp_rear_image)
            cv2.imshow("rgb-left-FOV-100", disp_left_image)
            cv2.imshow("rgb-right-FOV-100", disp_right_image)
            cv2.waitKey(1)

        return applied_control

    def dataset_save(self, rgb, measurement_data):
        for i in range(len(self.subfolder_paths) - 1):
            cv2.imwrite(os.path.join(self.subfolder_paths[i], "%04i.png" % self.data_count), rgb[i])
        
        with open(os.path.join(self.subfolder_paths[-1], "%04i.json" % self.data_count), 'w+', encoding='utf-8') as f:
            json.dump(measurement_data, f,  ensure_ascii=False, indent=4)

        self.data_count += 1

    def check_and_save(self, rgb, measurement_data):
        if self.red_wait_counter <= self.red_wait_limit:
            limit = 3
        else:
            limit = 20

        if self.save_counter >= limit:
            self.dataset_save(rgb, measurement_data)
            self.save_counter = 0
        else:
            self.save_counter += 1

    def check_dagger_metric(self, auto_pilot_control, dnn_agent_control):
        if(abs(auto_pilot_control.throttle - dnn_agent_control.throttle) > configs.throttle_metric_threshold or
           abs(auto_pilot_control.steer - dnn_agent_control.steer) > configs.steer_metric_threshold):

            self.dagger_counter += 1
            if self.dagger_counter >= 5:
                return True
            else:
                return False
        else:
            self.dagger_counter = 0
            return False
    
    def check_dagger_brake_metric(self, autopilot_brake, model_brake):
        if abs(autopilot_brake - model_brake) > configs.brake_metric_threshold:
            self.dagger_counter += 1

            if self.dagger_counter >= 3:
                return True
            else:
                return False
        
        else:
            self.dagger_counter = 0
            return False

    def _get_control(self, target, far_target, tick_data):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # steering
        angle_unnorm = base_utils.get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # acceleration
        angle_far_unnorm = base_utils.get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0

        target_speed = 4.0 if should_slow else 7.0

        if self.run_type is "autopilot" or self.run_type is "dagger" or self.run_type is "inference":
            brake = self._should_brake()
        else:
            brake = 0.0

        self.should_slow = int(should_slow)
        self.should_brake = int(brake)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm
        
        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        return steer, throttle, brake, target_speed

    def _change_weather(self):
        if self.weather_counter % self.weather_change_interval == 0:
            index = random.choice(range(len(base_utils.WEATHERS)))
            dtime, altitude = random.choice(list(base_utils.daytimes.items()))
            
            altitude = np.random.normal(altitude, 10)
            self.weather_id = base_utils.WEATHERS_IDS[index] + dtime

            weather = base_utils.WEATHERS[base_utils.WEATHERS_IDS[index]]
            weather.sun_altitude_angle = altitude
            weather.sun_azimuth_angle = np.random.choice(base_utils.azimuths)
            
            self._world.set_weather(weather)
        self.weather_counter += 1

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))
        stop_sign = self._is_stop_sign_hazard(actors.filter('*stop*'))

        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_red_light_present = 1 if light is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        self.is_stop_sign_present = 1 if stop_sign is not None else 0

        if self.is_red_light_present or self.is_vehicle_present:
            self.red_wait_counter += 1
        else: 
            self.red_wait_counter = 0

        return any(x is not None for x in [vehicle, light, walker, stop_sign])

    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _get_forward_speed(self, transform=None, velocity=None):
        
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        affected = False

        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._world.get_map().get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _is_stop_sign_hazard(self, stop_sign_list):
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return None
                else:
                    return self._target_stop_sign
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return None

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._world.get_map().get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    return self._target_stop_sign

        return None

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting
            
        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = base_utils._numpy(self._vehicle.get_location())
        v1 = 10.0 * base_utils._orientation(self._vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = base_utils._orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(base_utils._numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + base_utils._numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = base_utils.get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = base_utils._orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = base_utils._numpy(self._vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(base_utils._numpy(self._vehicle.get_velocity()))) # increases the threshold distance
        s2 = max(20, 3.0 * np.linalg.norm(base_utils._numpy(self._vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = base_utils._orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = base_utils._numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(base_utils._numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            # self.follow = False
            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1 and distance < s2:
                self.target_vehicle_speed = target_vehicle.get_velocity()
                # self.follow = True
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None
        