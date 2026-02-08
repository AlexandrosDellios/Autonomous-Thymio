import numpy as np

camera_port = 1

map_width = 100 # cm
map_height = 70 # cm
robot_width = 18 #12 # cm

max_retries = 5
epsilon_ratio=0.02
min_obstacle_area = 1500 # pixel
min_goal_area = 1500 # pixel


# 0 < blue, green, red < 255
# 0 < saturation, value < 255
# 0 < hue < 179

# map_th = {
#     "max_r": 180, "min_r": 0, "max_g": 130, "min_g": 0, "max_b": 160, "min_b": 0,
#     "max_h": 179, "min_h": 100, "max_s": 150, "min_s": 50, "max_v": 200, "min_v": 50,
# }
map_th = {
    "max_r": 255, "min_r": 140, "max_g": 255, "min_g": 150, "max_b": 255, "min_b": 130,
    "max_h": 150, "min_h": 0, "max_s": 80, "min_s": 0, "max_v": 255, "min_v": 150,
}

obstacle_th = {
    "max_r": 180, "min_r": 1, "max_g": 130, "min_g": 1, "max_b": 160, "min_b": 1,
    "max_h": 179, "min_h": 100, "max_s": 150, "min_s": 50, "max_v": 200, "min_v": 50,
}

goal_th = {
    "max_r": 255, "min_r": 150, "max_g": 160, "min_g": 50, "max_b": 220, "min_b": 100,
    "max_h": 180, "min_h": 150, "max_s": 175, "min_s": 100, "max_v": 255, "min_v": 220,
}

radius=5
thickness = -1
radius_pov_thymio = 2
thickness_line = 2
arrow_length = 50
red=(0, 0, 255)
green=(0, 255, 0)
blue=(255, 0, 0)

### Local navigation parameters
MOVING_AVERAGE_WINDOW_SIZE = 10 # We get data at 10Hz, so this is a 1 second moving average => it stays at least 1 second in local navigation
RADIUS_AVOIDANCE = 10 # Use to update to the next checkpoint if an obstacle is detected within this radius from the path
MAP_RES_X = 100  # number of cells in x direction
MAP_RES_Y = 70  # number of cells in y direction
RADIUS_INFLUENCE = 5  # radius of influence of absolute obstacles in cm
STRENGHT_INFLUENCE = 40  # maximum speed reduction due to obstacles
DISTANCE_KIDNAPPING = 20  # in cm, distance threshold to replan global path when a kidnapping is detected
DISTANCE_REPLAN = 20  # in cm, distance threshold to replan global path when close to next checkpoint
# Sensor positions and orientations
CENTER_OFFSET = np.array([5.5, 5.5])
SENSOR_POS = np.array([[0.8,9.4], [3.0,10.5], [5.5,11.0], [8.0,10.5], [10.2,9.4]])-CENTER_OFFSET
SENSOR_ANGLE = np.array([30, 15, 0, -15, -30])*np.pi/180

MAX_SPEED_CHECKPOINT_APPROACH = 100 # In thymio raw units

SPEED0 = 100 # Nominal speed for next checkpoint approach
SPEED1 = 0   # Nominal speed when close to global obstacle in local navigation
SPEED_GAIN = 2  # Gain used diff theta for next checkpoint approach
OBS_SPEED_GAIN = np.array([1, 0.6, -0.3, -0.6, -1])*10   # 0-1 gains used with front proximity sensors 0..4
