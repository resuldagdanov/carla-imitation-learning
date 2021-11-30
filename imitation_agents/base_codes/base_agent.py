import time
import os
import cv2
import carla
import numpy as np
from imitation_agents.base_codes import autonomous_agent
from imitation_agents.base_codes.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)


class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self._sensor_data = {
            'width': 400,
            'height': 300,
            'high_fov': 100,
            'mid_fov': 60,
            'low_fov': 45
        }

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self.initialized = True
        self._sensor_data['calibration'] = self._get_camera_to_car_calibration(self._sensor_data)
        self._sensors = self.sensor_interface._sensors_objects

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps

    def sensors(self):
        return [
                # {
                #     'type': 'sensor.camera.rgb',
                #     'x': 1.3, 'y': 0.0, 'z': 2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['high_fov'],
                #     'id': 'rgb_front_100'
                #     },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['mid_fov'],
                    'id': 'rgb_front_60'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['high_fov'],
                    'id': 'rgb_rear_100'
                    },
                # {
                #     'type': 'sensor.camera.rgb',
                #     'x': -1.3, 'y': 0.0, 'z': 2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['mid_fov'],
                #     'id': 'rgb_rear_60'
                #     },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['high_fov'],
                    'id': 'rgb_left_100'
                    },
                # {
                #     'type': 'sensor.camera.rgb',
                #     'x': 1.3, 'y': 0.0, 'z': 2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['mid_fov'],
                #     'id': 'rgb_left_60'
                #     },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['high_fov'],
                    'id': 'rgb_right_100'
                    },
                # {
                #     'type': 'sensor.camera.rgb',
                #     'x': 1.3, 'y': 0.0, 'z': 2.3,
                #     'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['mid_fov'],
                #     'id': 'rgb_right_60'
                #     },
                {   
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'id': 'lidar'
                    },
                {
                    'type': 'sensor.other.radar',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'fov': 30, 'range': 50,
                    'id': 'radar'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def tick(self, input_data):
        self.step += 1

        affordances = self._get_affordances()

        # rgb_front_100 = cv2.cvtColor(input_data['rgb_front_100'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front_60 = cv2.cvtColor(input_data['rgb_front_60'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        rgb_rear_100 = cv2.cvtColor(input_data['rgb_rear_100'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        # rgb_rear_60 = cv2.cvtColor(input_data['rgb_rear_60'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        rgb_left_100 = cv2.cvtColor(input_data['rgb_left_100'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        # rgb_left_60 = cv2.cvtColor(input_data['rgb_left_60'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        rgb_right_100 = cv2.cvtColor(input_data['rgb_right_100'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        # rgb_right_60 = cv2.cvtColor(input_data['rgb_right_60'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        lidar = input_data['lidar'][1]
        radar = input_data['radar'][1]

        weather = self._weather_to_dict(self._world.get_weather())

        return {
            # 'rgb_front_100': rgb_front_100,
            'rgb_front_60': rgb_front_60,

            'rgb_rear_100': rgb_rear_100,
            # 'rgb_rear_60': rgb_rear_60,
            
            'rgb_left_100': rgb_left_100,
            # 'rgb_left_60': rgb_left_60,

            'rgb_right_100': rgb_right_100,
            # 'rgb_right_60': rgb_right_60,

            'lidar' : lidar,
            'radar' : radar,

            'gps': gps,
            'speed': speed,
            'compass': compass,

            'weather': weather,
            'affordances': affordances,
            }
            
    def _weather_to_dict(self, carla_weather):
        weather = {
            'cloudiness': carla_weather.cloudiness,
            'precipitation': carla_weather.precipitation,
            'precipitation_deposits': carla_weather.precipitation_deposits,
            'wind_intensity': carla_weather.wind_intensity,
            'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
            'sun_altitude_angle': carla_weather.sun_altitude_angle,
            'fog_density': carla_weather.fog_density,
            'fog_distance': carla_weather.fog_distance,
            'wetness': carla_weather.wetness,
            'fog_falloff': carla_weather.fog_falloff,
        }

        return weather

    def _translate_tl_state(self, state):

        if state == carla.TrafficLightState.Red:
            return 0
        elif state == carla.TrafficLightState.Yellow:
            return 1
        elif state == carla.TrafficLightState.Green:
            return 2
        elif state == carla.TrafficLightState.Off:
            return 3
        elif state == carla.TrafficLightState.Unknown:
            return 4
        else:
            return None

    def _get_affordances(self):
        
        # affordance tl
        affordances = {}
        affordances["traffic_light"] = None
        
        affecting = self._vehicle.get_traffic_light()

        if affecting is not None:
            for light in self._traffic_lights:
                if light.id == affecting.id:
                    affordances["traffic_light"] = self._translate_tl_state(self._vehicle.get_traffic_light_state())

        affordances["stop_sign"] = self._affected_by_stop

        return affordances

    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def _change_seg_tl(self, seg_img, depth_img, traffic_lights, _region_size=4):
        """Adds 3 traffic light classes (green, yellow, red) to the segmentation image

        Args:
            seg_img ([type]): [description]
            depth_img ([type]): [description]
            traffic_lights ([type]): [description]
            _region_size (int, optional): [description]. Defaults to 4.
        """        
        for tl in traffic_lights:
            _dist = self._get_distance(tl.get_transform().location)
                
            _region = np.abs(depth_img - _dist)

            if tl.get_state() == carla.TrafficLightState.Red:
                state = 23
            elif tl.get_state() == carla.TrafficLightState.Yellow:
                state = 24
            elif tl.get_state() == carla.TrafficLightState.Green:
                state = 25
            else: #none of the states above, do not change class
                state = 18

            #seg_img[(_region >= _region_size)] = 0
            seg_img[(_region < _region_size) & (seg_img == 18)] = state

    def _get_dist(self, p1, p2):
        """Returns the distance between p1 and p2

        Args:
            target ([type]): [description]

        Returns:
            [type]: [description]
        """        

        distance = np.sqrt(
                (p1[0] - p2[0]) ** 2 +
                (p1[1] - p2[1]) ** 2 +
                (p1[2] - p2[2]) ** 2)

        return distance

    def _get_distance(self, target):
        """Returns the distance from the (rgb_front) camera to the target

        Args:
            target ([type]): [description]

        Returns:
            [type]: [description]
        """        
        sensor_transform = self._sensors['rgb_front'].get_transform()

        distance = np.sqrt(
                (sensor_transform.location.x - target.x) ** 2 +
                (sensor_transform.location.y - target.y) ** 2 +
                (sensor_transform.location.z - target.z) ** 2)

        return distance

    def _get_depth(self, data):
        """Transforms the depth image into meters

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """        

        data = data.astype(np.float32)

        normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
        normalized /=  (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized

        return in_meters

    def _find_obstacle(self, obstacle_type='*traffic_light*'):
        """Find all actors of a certain type that are close to the vehicle

        Args:
            obstacle_type (str, optional): [description]. Defaults to '*traffic_light*'.

        Returns:
            [type]: [description]
        """        
        obst = list()
        
        _actors = self._world.get_actors()
        _obstacles = _actors.filter(obstacle_type)


        for _obstacle in _obstacles:
            trigger = _obstacle.trigger_volume

            _obstacle.get_transform().transform(trigger.location)
            
            distance_to_car = trigger.location.distance(self._vehicle.get_location())

            a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
            b = np.sqrt(
                self._vehicle.bounding_box.extent.x ** 2 +
                self._vehicle.bounding_box.extent.y ** 2 +
                self._vehicle.bounding_box.extent.z ** 2)

            s = a + b + 10
           

            if distance_to_car <= s:
                # the actor is affected by this obstacle.
                obst.append(_obstacle)

        return obst

    def _get_camera_to_car_calibration(self, sensor):
        """returns the calibration matrix for the given sensor

        Args:
            sensor ([type]): [description]

        Returns:
            [type]: [description]
        """        
        calibration = np.identity(3)
        calibration[0, 2] = sensor["width"] / 2.0
        calibration[1, 2] = sensor["height"] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor["width"] / (2.0 * np.tan(sensor["mid_fov"] * np.pi / 360.0))
        return calibration
