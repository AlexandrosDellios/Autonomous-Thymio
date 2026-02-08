from tdmclient import ClientAsync, aw
import asyncio

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import time
import keyboard
import cv2

import vision as vision
import config as config

from aayushi_path_planning import construct_visibility_graph, plan_global_navigation, compute_orientations, plot_visibility_graph
from kalman_filter import ThymioEKF

smoothed_distance = [[0,0,0,0,0] for _ in range(config.MOVING_AVERAGE_WINDOW_SIZE)]

def current_position_update_thymio(current_pos,motor_speed_left,motor_speed_right, dt):
    raw_to_velocity = 21/11.29/50  # conversion factor from raw motor speed to cm/s
    wheel_separation=9.5 #measure
    v_left = motor_speed_left * raw_to_velocity
    v_right = motor_speed_right * raw_to_velocity
    v = (v_right + v_left) / 2
    omega = (v_right - v_left) / wheel_separation
    dx = v * np.cos(current_pos[2]) * dt
    dy = v * np.sin(current_pos[2]) * dt
    dtheta = omega * dt
    updated_pos = [current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dtheta]
    return updated_pos

def update_local_grid(local_grid,current_pos, prox_horizontal_i,i,path):
    # Calculate grid position (separate variables instead of tuple)
    pos_x=current_pos[0]
    pos_y=current_pos[1]
    grid_pos_x = pos_x * config.MAP_RES_X // config.map_width
    grid_pos_y = pos_y * config.MAP_RES_Y // config.map_height
    IS_OBSTACLE_DETECTED_IN_PATH = False

    # from 5 to 12 cm obstacle distance, on the angle of the sensor, on the basis frame reference,
    # we update the local grid to 1 to the first cell corresponding to the obstacle position
    # else we put 0 to the line, following the sensor direction
    sensor_angle = config.SENSOR_ANGLE[i] + current_pos[2]
    sensor_pos = config.SENSOR_POS[i] + np.array([pos_x, pos_y])
    if prox_horizontal_i > 0 :
        distance = (5739.9 - prox_horizontal_i) / 339.47
        if distance >= 5 and distance <= 12:
            obstacle_x = sensor_pos[0] + distance * np.cos(sensor_angle)
            obstacle_y = sensor_pos[1] + distance * np.sin(sensor_angle)
            for checkpoints in path:
                if np.sqrt((obstacle_x - checkpoints[0])**2 + (obstacle_y - checkpoints[1])**2) < config.RADIUS_AVOIDANCE:
                    IS_OBSTACLE_DETECTED_IN_PATH = True
            grid_obstacle_x = int(obstacle_x * config.MAP_RES_X // config.map_width)
            grid_obstacle_y = int(obstacle_y * config.MAP_RES_Y // config.map_height)
            if 0 <= grid_obstacle_x < config.MAP_RES_X and 0 <= grid_obstacle_y < config.MAP_RES_Y:
                # print(type(local_grid))
                local_grid[grid_obstacle_x, grid_obstacle_y] = 1
    else:
        # No obstacle detected in range, clear the line in the direction of the sensor between 0 and 15cm for each cell in between
        clear_points = np.array([[j * np.cos(sensor_angle) + sensor_pos[0],
                                  j * np.sin(sensor_angle) + sensor_pos[1]] for j in range(0, 16)])
        grid_clear_points = np.array(clear_points * np.array([config.MAP_RES_X / config.map_width, config.MAP_RES_Y / config.map_height]), dtype=int)
        for points in grid_clear_points:
            if not (0 <= points[0] < config.MAP_RES_X and 0 <= points[1] < config.MAP_RES_Y):
                grid_clear_points = grid_clear_points[grid_clear_points[:, 0] < config.MAP_RES_X]
                grid_clear_points = grid_clear_points[grid_clear_points[:, 1] < config.MAP_RES_Y]
                grid_clear_points = grid_clear_points[grid_clear_points[:, 0] >= 0]
                grid_clear_points = grid_clear_points[grid_clear_points[:, 1] >= 0]
                break
            else:
                # reset points around the sensors because of sensor noise
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx = points[0] + dx
                        ny = points[1] + dy
                        if 0 <= nx < config.MAP_RES_X and 0 <= ny < config.MAP_RES_Y:
                            local_grid[nx, ny] = 0
    return IS_OBSTACLE_DETECTED_IN_PATH, local_grid

def local_navigation(next_checkpoints, current_pos, distance, obstacle_map):
    # basic speed might need to be tune next to a checkpoint
    # maximum of 100 speed to avoid max speed toward checqkpoint if far away

    pos_x=current_pos[0]
    pos_y=current_pos[1]
    pos_theta=current_pos[2]
    diffDelta = np.arctan2(next_checkpoints[1] - pos_y, next_checkpoints[0] - pos_x) - pos_theta
    if diffDelta > 3.14:
        diffDelta -= 2 * 3.14
    elif diffDelta < -3.14:
        diffDelta += 2 * 3.14
    # speed based on nominal velocity and ground (gradient due to goal)
    spLeft = config.SPEED0 - config.SPEED_GAIN * diffDelta
    spRight = config.SPEED0 + config.SPEED_GAIN * diffDelta
    # print("spLeft before limit:", spLeft)
    # print("spRight before limit:", spRight)

    # We don't allow more than 100 speed for checkpoint approach and there's no max function
    spMax = np.max([abs(spLeft), abs(spRight)])

    if spMax > config.MAX_SPEED_CHECKPOINT_APPROACH or spRight > config.MAX_SPEED_CHECKPOINT_APPROACH:
        spLeft = spLeft * config.MAX_SPEED_CHECKPOINT_APPROACH // spMax
        spRight = spRight * config.MAX_SPEED_CHECKPOINT_APPROACH // spMax

    # adjustment for obstacles ("gradient" due to obstacles)
    # spLeft, spRight = 0,0
    for i in range(5):
        spLeft += (12 - distance[i]) * config.OBS_SPEED_GAIN[i]
        spRight += (12 - distance[i]) * config.OBS_SPEED_GAIN[4 - i]
    #     print("Distance sensor", i, ":", distance[i])
    #     print("spLeft intermediate for sensor", i, ":", spLeft)
    # print("spLeft after obstacle adjustment:", spLeft)
    # print("spRight after obstacle adjustment:", spRight)


    # Calculate grid position (separate variables instead of tuple)
    grid_pos_x = int(pos_x * config.MAP_RES_X // config.map_width)
    grid_pos_y = int(pos_y * config.MAP_RES_Y // config.map_height)

    # compute the positions of the obstacles that will have an influence on the robot with the config.RADIUS_INFLUENCE
    for i in range(-config.RADIUS_INFLUENCE, config.RADIUS_INFLUENCE + 1):
        for j in range(-config.RADIUS_INFLUENCE, config.RADIUS_INFLUENCE + 1):
            check_x = grid_pos_x + i
            check_y = grid_pos_y + j
            if 0 <= check_x < config.MAP_RES_X and 0 <= check_y < config.MAP_RES_Y:
                # Calculate distance squared (avoid sqrt and float division)
                dist_sq = i * i + j * j
                if dist_sq <= config.RADIUS_INFLUENCE * config.RADIUS_INFLUENCE:
                    if obstacle_map[check_x, check_y] == 1:
                        # sg = 1 if the obstacle is on the left side, -1 if on the right side
                        sg = 1 if (i * np.cos(pos_theta) + j * np.sin(pos_theta)) > 0 else -1
                        # Simplified influence calculation (avoid float division)
                        influence = max(0, (config.RADIUS_INFLUENCE - np.sqrt(dist_sq))/config.RADIUS_INFLUENCE)
                        spLeft += config.SPEED1 + sg * config.STRENGHT_INFLUENCE * influence
                        spRight += config.SPEED1 - sg * config.STRENGHT_INFLUENCE * influence

    # motor control
    motor_left_target = spLeft
    motor_right_target = spRight
    return motor_left_target, motor_right_target

def correct_orientation(target_or, current_or, Kp=30):
    angle=target_or-current_or
    if angle>3.14:
        angle-=2*3.14
    elif angle<-3.14:
        angle+=2*3.14
    speed=Kp*angle
    #if angle is positive, we want anticlockwise motion- left motor backwards, right motor forwards
    motor_left_target=-speed
    motor_right_target=speed
    return motor_left_target, motor_right_target, 0

def straight_line_motion(target,current_pos, Kp=10): #Epsilon 5 cm  for uncertainty as its approx half the width of the robot
    eps_dist=1 #tune this value for the P controller
    #assume orientation has already been corrected before this step
    dist=np.sqrt((target[0]-current_pos[0])**2+(target[1]-current_pos[1])**2)
    speed=Kp*dist
    thymio_heading_vector = np.array([np.cos(current_pos[2]), np.sin(current_pos[2])])
    displacement_vector = np.array([target[0]-current_pos[0], target[1]-current_pos[1]])
    sign_correction = np.sign(np.dot(thymio_heading_vector, displacement_vector))
    motor_left_target = sign_correction*speed
    motor_right_target = sign_correction*speed
    if abs(dist)<=eps_dist:
        #return 1 to signify current target node is reached - switch the target node to the next in the path
        #return 0 to signify we still need to move towards the same target node
        return motor_left_target, motor_right_target, 1
    return motor_left_target, motor_right_target, 0

def global_navigation(target,current_pos):
    #Decide whether to correct orientation or move in straight line
    angle = np.arctan2(target[1]-current_pos[1],target[0]-current_pos[0])
    angle_diff=angle-current_pos[2]
    dist=np.sqrt((target[0]-current_pos[0])**2+(target[1]-current_pos[1])**2)
    eps_theta=0.087 #5 degrees in radians
    eps_dist=1 #5 cm
    if abs(angle_diff)<=eps_theta and dist>=eps_dist:
        return straight_line_motion(target,current_pos)
    if dist>=eps_dist and abs(angle_diff)>=eps_theta:
        return correct_orientation(angle,current_pos[2])
    #Correct qngle first if both distance and angle error are significant
    if abs(angle_diff)>=eps_theta and dist<=eps_dist:
        return (0,0,1)
    else:
        return (0,0,1)

def raw_to_velocity(raw_vel_1):
    return 0.0268*raw_vel_1 + 0.4754 # Convert motor command to cm/s

def update_moving_average(distance,window_size,i):
    # Apply moving average filter to smooth distance readings
    global smoothed_distance
    smoothed_distance[i].append(distance)
    if len(smoothed_distance[i]) > window_size:
        smoothed_distance[i].pop(0)
    return np.mean(smoothed_distance[i])

async def main():
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    Node = client.nodes[0]

async def thymiodata():
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    Node = client.nodes[0]
    aw(Node.lock())
    await Node.wait_for_variables({"prox.horizontal","motor.right.speed","motor.left.speed"})
    prox_horizontal=list(Node.v.prox.horizontal)
    motor_right_speed=Node.v.motor.right.speed
    motor_left_speed=Node.v.motor.left.speed
    return prox_horizontal, motor_right_speed, motor_left_speed, Node

#Get obstacle coordinates from camera
if __name__ == "__main__":
    # asyncio.run(main())
    # obstacle_map[7][1] = 0  # example obstacle
    local_grid = np.zeros((config.MAP_RES_X, config.MAP_RES_Y))
    cam = cv2.VideoCapture(config.camera_port, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise ValueError("Camera failed to open")

    time.sleep(1)

    M, width, height = vision.retry_detection(
        "Map",
        vision.reframe_rect,
        cam
    )
    pixel_size = vision.pixel_to_cm(cam, M, width, height)
    obstacle_points = vision.retry_detection(
        "Obstacles",
        vision.get_obstacles,
        cam,
        M=M, width=width, height=height, pixel_size=pixel_size
    )[0]
    goal_pos = vision.retry_detection(
        "Goal",
        vision.goal_detection,
        cam,
        M=M, width=width, height=height, pixel_size=pixel_size
    )[0]
    robot_pos = vision.retry_detection(
        "Robot",
        vision.compute_robot_pos,
        cam,
        raise_on_fail=False,
        fallback=(np.array((None, None, None))),
        M=M, width=width, height=height,obstacle_points=obstacle_points, goal_pos=goal_pos, pixel_size=pixel_size, path=None
    )[0]
    raw_obstacles = vision.format_obstacles(obstacle_points) # output for global nav
    obstacle_mask = vision.mask_from_points(raw_obstacles) # output for local nav

    print("robot pos: ", robot_pos)
    start=robot_pos[:2]
    print("start: ", start)
    goal=goal_pos
    G=construct_visibility_graph(raw_obstacles, start, goal,(0,config.map_width,0,config.map_height))
    path_idx=plan_global_navigation(G,'S','G')
    path=compute_orientations(G,path_idx)
    # print("Planned path (x,y,theta): ", path)
    plot_visibility_graph(G, raw_obstacles, start, goal, path)
    # path = get_path()
    # path = [[0,0,0],[10,0,0]]

    checkpoint_index=1
    next_checkpoint=path[checkpoint_index]

    current_time=time.time()
    local_grid = np.zeros((config.MAP_RES_X, config.MAP_RES_Y))

    iterate = 0
    ekf =  ThymioEKF()

    ekf.initialize_kalman_pos(robot_pos)
    cv2.namedWindow("Thymio POV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Thymio POV", 500, 350)
    vision.draw_Thymio_POV(obstacle_mask, local_grid, robot_pos)
    last_camera_pos=robot_pos
    local_distance_from_trajectory = 0
    while True:
        robot_pos = vision.retry_detection(
            "Robot",
            vision.compute_robot_pos,
            cam,
            raise_on_fail=False,
            fallback=(np.array((None, None, None))),
            M=M, width=width, height=height,obstacle_points=obstacle_points, goal_pos=goal_pos, pixel_size=pixel_size, path=path
        )[0]
        new_time=time.time()
        prox_horizontal, motor_right_speed, motor_left_speed, Node = asyncio.run(thymiodata())
        dt=new_time-current_time
        motor_left_speed_cm_s = raw_to_velocity(motor_left_speed)
        motor_right_speed_cm_s = raw_to_velocity(motor_right_speed)
        ekf.predict_Jacob(dt,np.array([motor_left_speed_cm_s,motor_right_speed_cm_s]))
        ekf.update_modified(robot_pos,ekf.H_jacobian,ekf.hx,ekf.R)
        current_pos=ekf.x
        if robot_pos is not None:
            if np.linalg.norm(np.array(robot_pos[0:2]) - np.array(last_camera_pos[0:2])) > config.DISTANCE_KIDNAPPING:
                G=construct_visibility_graph(raw_obstacles, robot_pos[:2], goal,(0,config.map_width,0,config.map_height))
                path_idx=plan_global_navigation(G,'S','G')
                path=compute_orientations(G,path_idx)
                checkpoint_index=1
                next_checkpoint=path[checkpoint_index]
            last_camera_pos=robot_pos
        vision.draw_Thymio_POV(obstacle_mask, local_grid, current_pos)

        if current_pos.any() == None:
            print("robot pos ekf is NONE")
        print("Current pos :",current_pos)
        local=False
        distance=[0,0,0,0,0]
        for i in range(5):
            if prox_horizontal[i]==0:
                distance[i]=12
                continue
            distance[i]=(5739.9-prox_horizontal[i])/339.47
            distance[i] = update_moving_average(distance[i], config.MOVING_AVERAGE_WINDOW_SIZE, i)
            if distance[i]>=0 and distance[i]<12:
                local=True #Flag used to not continue global navigation right after local, but to wait till all sensor values are clear
            IS_OBSTACLE_IN_PATH, local_grid = update_local_grid(local_grid, current_pos,prox_horizontal[i],i,path)
            if IS_OBSTACLE_IN_PATH:
                if checkpoint_index != len(path)-1:
                    checkpoint_index+=1
                    next_checkpoint=path[checkpoint_index]
                else :
                    if checkpoint_index>1:
                        checkpoint_index-=1
                        next_checkpoint=path[checkpoint_index]
        print("am i in local ?", local)
        if local:
            local_distance_from_trajectory = np.linalg.norm(np.array(current_pos[0:2]) - np.array(next_checkpoint[0:2]))
            if local_distance_from_trajectory > config.DISTANCE_REPLAN:
                G=construct_visibility_graph(raw_obstacles, current_pos[:2], goal,(0,config.map_width,0,config.map_height))
                G.remove_node(checkpoint_index) #Update the start node to current position
                path_idx=plan_global_navigation(G,'S','G')
                path=compute_orientations(G,path_idx)
                checkpoint_index=1
                next_checkpoint=path[checkpoint_index]
            motor_left_speed, motor_right_speed = local_navigation(next_checkpoint, current_pos, distance, obstacle_mask.T)
        else :
            motor_left_speed, motor_right_speed, reached = global_navigation(next_checkpoint,current_pos)
            if reached==1:
                print("Reached")
                checkpoint_index+=1
                accumulated_error_theta = 0
                if checkpoint_index>=len(path):
                    motor_left_speed=0
                    motor_right_speed=0
                    v = {
                        "motor.left.target": [int(motor_left_speed)],
                        "motor.right.target": [int(motor_right_speed)],
                    }
                    aw(Node.set_variables(v))

                    break
                next_checkpoint=path[checkpoint_index]


        v = {
            "motor.left.target": [int(motor_left_speed)],
            "motor.right.target": [int(motor_right_speed)],
        }
        aw(Node.set_variables(v))

        #Stop the loop outside the lock/unlock to avoid deadlock
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to break
            print("Loop stopped by user.")
            v = {
            "motor.left.target": [int(0)],
            "motor.right.target": [int(0)],
            }
            aw(Node.set_variables(v))
            aw(Node.unlock())
            break
        aw(Node.unlock())
        current_time=new_time

    cam.release()
    cv2.destroyAllWindows()
