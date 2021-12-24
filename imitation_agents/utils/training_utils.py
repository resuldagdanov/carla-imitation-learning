import numpy as np


def scale_and_crop_image(image, scale=1, crop=256):
    (width, height) = (int(image.width // scale), int(image.height // scale))

    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)

    start_x = height//2 - crop//2
    start_y = width//2 - crop//2

    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    cropped_image = np.transpose(cropped_image, (2,0,1))

    return cropped_image


def lidar_to_histogram_features(lidar, crop=256):
    # 256 x 256 grid

    def splat_points(point_cloud):
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32

        xbins = np.linspace(-2*x_meters_max, 2*x_meters_max+1, 2*x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)

        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel

        overhead_splat = hist/hist_max_per_pixel

        return overhead_splat

    below = lidar[lidar[...,2] <= 2]
    above = lidar[lidar[...,2] > 2]

    below_features = splat_points(below)
    above_features = splat_points(above)
    
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)

    return features


def calculate_action_loss(throttle, steer, brake, gt, criterion_throttle, criterion_steer, criterion_brake):
    throttle_loss = criterion_throttle(throttle, gt[:, 0].view(-1).long())
    steer_loss = criterion_steer(steer, gt[:, 1].view(-1, 1))
    brake_loss = criterion_brake(brake, gt[:, 2].view(-1, 1))

    return throttle_loss + steer_loss + brake_loss


def clip_throttle(throttle):
    if throttle <= 0.05:
        return 0

    elif throttle > 0.05 and throttle < 0.4:
        return 1

    else:
        return 2
    

def clip_steer(steer):
    if steer <= -0.3:
        return 0

    elif steer > -0.3 and steer < -0.1:
        return 1

    elif steer > -0.1 and steer < -0.02:
        return 2

    elif steer > -0.02 and steer < -0.005:
        return 3

    elif steer >= -0.005 and steer <= 0.005:
        return 4

    elif steer > 0.005 and steer < 0.02:
        return 5

    elif steer > 0.02 and steer < 0.1:
        return 6

    elif steer > 0.1 and steer < 0.3:
        return 7

    elif steer >= 0.3:
        return 8

    else:
        return steer
