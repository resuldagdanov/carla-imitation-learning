import carla
import numpy as np


WEATHERS = {
		'Clear': carla.WeatherParameters.ClearNoon,

		'Cloudy': carla.WeatherParameters.CloudySunset,

		'Wet': carla.WeatherParameters.WetSunset,

		'MidRain': carla.WeatherParameters.MidRainSunset,

		'WetCloudy': carla.WeatherParameters.WetCloudySunset,

		'HardRain': carla.WeatherParameters.HardRainNoon,

		'SoftRain': carla.WeatherParameters.SoftRainSunset,
}


WEATHERS_IDS = list(WEATHERS)


azimuths = [45.0 * i for i in range(8)]


daytimes = {
	'Night': -35.0,
	'Twilight': 0.0,
	'Dawn': 5.0,
	'Sunset': 15.0,
	'Morning': 35.0,
	'Noon': 75.0,
}


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def get_angle_to(pos, theta, target):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    aim = R.T.dot(target - pos)
    
    angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
    angle = 0.0 if np.isnan(angle) else angle 

    return angle


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

    return collides, p1 + x[0] * v1
