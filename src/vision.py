import cv2
import numpy as np
import scipy

import config as config


def grab_frame(cam):
    for i in range(config.max_retries):
        ret, frame = cam.read()
        if ret:
            break
        print(f"Camera failed to grab frame ({i+1}/{config.max_retries}) times", end="\r")
        if i == 4:
            raise ValueError("Camera failed to grab frame too many times")
    return frame

def retry_detection(
    name,
    func,
    cam,
    raise_on_fail=True,
    fallback=None,
    *args, **kwargs
):
    """
    Generic retry wrapper for detection-like functions.

    Assumes func(frame, *args, **kwargs) returns (*data, detected)
    """
    for i in range(config.max_retries):
        frame = grab_frame(cam)
        result = func(frame, *args, **kwargs)
        *data, detected = result

        if detected:
            # print(f"{name} detected!")
            return data

        print(f"{name} not detected ({i+1}/{config.max_retries}) times", end="\r")

    if raise_on_fail:
        raise ValueError(f"{name} was not detected properly too many times")
    else:
        print(f"{name} NOT detected, using fallback.")
        return fallback

def extract_channels(img):
    data_0 = img[:, :, 0]
    data_1 = img[:, :, 1]
    data_2 = img[:, :, 2]

    return data_0, data_1, data_2

def detect_mask(img, th):
    h, w, _ = np.shape(img)
    mask = np.zeros((h, w))

    data_b, data_g, data_r = extract_channels(img)
    data_h, data_s, data_v = extract_channels(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    mask = ((data_r <= th["max_r"]) & (data_r >= th["min_r"]) &
            (data_g <= th["max_g"]) & (data_g >= th["min_g"]) &
            (data_b <= th["max_b"]) & (data_b >= th["min_b"]) &
            (data_h <= th["max_h"]) & (data_h >= th["min_h"]) &
            (data_s <= th["max_s"]) & (data_s >= th["min_s"]) &
            (data_v <= th["max_v"]) & (data_v >= th["min_v"]))
    mask = (mask > 0).astype(np.uint8) * 255
    return mask

def get_rect_corner(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, False

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) != 4:
        # print(f"Expected 4 points, got {len(approx)} — mask might not be a rectangle")
        return None, False

    pts = approx.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2)).astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]

    return rect, True

def reframe_rect(frame):
    mask = detect_mask(frame, config.map_th)
    cv2.imshow("Map", mask)

    mask_filled = scipy.ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
    rect, detected = get_rect_corner(mask_filled)
    if not detected:
        return None, None, None, False

    width_top = np.linalg.norm(rect[0] - rect[1])
    width_bottom = np.linalg.norm(rect[3] - rect[2])
    width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(rect[0] - rect[3])
    height_right = np.linalg.norm(rect[1] - rect[2])
    height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)

    return M, width, height, True

def get_obstacles(frame, M, width, height, pixel_size):
    processed_frame = cv2.warpPerspective(frame, M, (width, height))
    mask = detect_mask(processed_frame, config.obstacle_th)
    cv2.imshow("Obstacles", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    offset=config.robot_width/2
    obstacles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < config.min_obstacle_area:
            continue

        peri = cv2.arcLength(cnt, True)
        eps = config.epsilon_ratio * peri

        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts = pts * pixel_size

        offset_pts = []
        for i in range(len(pts)):
            p_prev = pts[(i - 1) % len(pts)]
            p_curr = pts[i]
            p_next = pts[(i + 1) % len(pts)]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            v1 /= np.linalg.norm(v1) + 1e-6
            v2 /= np.linalg.norm(v2) + 1e-6

            bisector = v1 + v2
            norm = np.linalg.norm(bisector)
            if norm < 1e-6:
                bisector = v1
            else:
                bisector /= norm

            new_pt = p_curr - bisector * offset
            offset_pts.append(new_pt)
        obstacles.append(np.array(offset_pts, dtype=np.float32))

    if not obstacles:
        return None, False

    return obstacles, True

def format_obstacles(obstacle_points):
    rows = []
    for idx, coords in enumerate(obstacle_points):
        for x, y in coords:
            y = config.map_height - y
            rows.append([x, y, idx])
    arr = np.array(rows)
    return arr

def mask_from_points(obstacle_array):
    mask = np.zeros((config.map_height, config.map_width), dtype=np.uint8)

    object_ids = np.unique(obstacle_array[:, 2])

    for obj_id in object_ids:
        points = obstacle_array[obstacle_array[:, 2] == obj_id][:, :2]

        pts = np.round(points).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [pts], 1)
    return mask

def draw_obstacles(frame, obstacle_points, pixel_size):
    for obstacle in obstacle_points:
        obstacle = obstacle/pixel_size
        for (x, y) in np.int32(obstacle):
            cv2.circle(frame, (x, y), config.radius, config.blue, config.thickness)
    return frame

def goal_detection(frame, M, width, height, pixel_size):
    processed_frame = cv2.warpPerspective(frame, M, (width, height))
    mask = detect_mask(processed_frame, config.goal_th)
    cv2.imshow("Goal", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours :
        return None, False

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < config.min_goal_area:
        return None, False

    peri = cv2.arcLength(cnt, True)
    epsilon = config.epsilon_ratio * peri
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) != 4:
        # print(f"Expected 4 points, got {len(approx)} — mask might not be a rectangle")
        return None, False

    corners_goal = approx.reshape(-1, 2).astype(np.float32)
    center = corners_goal.mean(axis=0) * pixel_size
    center[1] = config.map_height - center[1]
    return center, True

def draw_point(frame, point, pixel_size):
    pt = point.copy()
    pt[1] = config.map_height - pt[1]
    pt = pt/pixel_size
    cv2.circle(frame, np.int32(pt), config.radius, config.green, config.thickness)
    return frame

def get_marker_corners(frame, pixel_size):
    markerCorners = []
    markerIds = []
    detectorParams = cv2.aruco.DetectorParameters()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    markerCorners, markerIds, _ = detector.detectMarkers(frame)

    if not markerIds:
        return None, False
    return np.array(markerCorners[0][0])*pixel_size, True

def compute_robot_pos(frame, M, width, height, obstacle_points, goal_pos, pixel_size, path):
    processed_frame = cv2.warpPerspective(frame, M, (width, height))
    robot_corners, detected = get_marker_corners(processed_frame, pixel_size)
    if not detected:
        draw(frame, M, width, height, obstacle_points, goal_pos, None, pixel_size, path)
        return None, False

    center = robot_corners.mean(axis=0)

    v = robot_corners[1] - robot_corners[0]
    v = v / np.linalg.norm(v)
    angles = np.arctan2(v[1], v[0])

    pos = ((center[0], config.map_height-center[1], -angles))
    draw(frame, M, width, height, obstacle_points, goal_pos, pos, pixel_size, path)
    return pos, True

def draw_robot(frame, pos, pixel_size):
    if not pos:
        return frame
    center = np.array(pos[:2])
    center[1] = config.map_height - center[1]
    center /= pixel_size
    end_x = int(center[0] + config.arrow_length * np.cos(pos[2]))
    end_y = int(center[1] + config.arrow_length * np.sin(-pos[2]))
    end_point = np.array((end_x, end_y))

    cv2.circle(frame, np.int32(center), config.radius, config.red, config.thickness)
    cv2.arrowedLine(frame,
        tuple(center.astype(int)),
        tuple(end_point.astype(int)),
        config.red, 2)
    return frame

def draw(frame, M, width, height, obstacle_points, goal_pos, robot_pos, pixel_size, path):
    cv2.imshow('Raw camera', frame)
    processed_frame = cv2.warpPerspective(frame, M, (width, height))
    processed_frame = draw_path(processed_frame, path, pixel_size)
    processed_frame = draw_obstacles(processed_frame, obstacle_points, pixel_size)
    processed_frame = draw_point(processed_frame, goal_pos, pixel_size)
    processed_frame = draw_robot(processed_frame, robot_pos, pixel_size)
    cv2.imshow('Camera', processed_frame)
    return

def draw_path(frame, path, pixel_size):
    if path==None:
        return frame
    for i in range(len(path)-1):
        coord = (int(path[i][0] / pixel_size[0]), int((config.map_height-path[i][1]) / pixel_size[1]))
        coord_next = (int(path[i+1][0] / pixel_size[0]), int((config.map_height-path[i+1][1]) / pixel_size[1]))
        cv2.line(frame, np.int32(coord), np.int32(coord_next), config.blue, thickness=config.thickness_line)
    return frame

def pixel_to_cm(cam, M, width, height):
    frame = grab_frame(cam)
    processed_frame = cv2.warpPerspective(frame, M, (width, height))
    pixel_size = (config.map_width/len(processed_frame[0]), config.map_height/len(processed_frame))
    return pixel_size

def draw_Thymio_POV(obstacle_mask, local_mask, thymio_pos):
    # obstacle_mask = obstacle_mask.T
    local_mask = local_mask.T
    obstacle_mask = obstacle_mask[::-1, :]# Flip vertically to match POV
    img = np.zeros((obstacle_mask.shape[0], obstacle_mask.shape[1], 3), dtype=np.uint8)
    img[obstacle_mask > 0] = config.blue
    img[local_mask > 0] = config.green
    pos = (thymio_pos[0], config.map_height-thymio_pos[1])
    cv2.circle(img, np.int32(pos), config.radius_pov_thymio, config.red, config.thickness)
    cv2.imshow("Thymio POV", img)
    return


