import os
from os.path import join
import cv2
import numpy as np

import utils
import constants as c
import masks

import matplotlib.pyplot as plt

from cars import *
from hog import *

from sklearn.svm import LinearSVC


# Define a class to store data from video
class Vehicle_Detect():
    def __init__(self):
        # history of detections from previous n frames
        self.prev_detects = [] 
        
    def add_detects(self, detects):
        self.prev_detects.append(detects)
        if len(self.prev_detects) > 15:
            # dismiss oldest detections
            self.prev_detects = self.prev_detects[len(self.prev_detects)-15:]

def initialize():
    print('CAMERA CALIBRATION')
    [camera_mtx, dist_params] = utils.calibrate_camera()

    print('TRAIN CLASSIFIER')
    scaler, model = train_model()

    return camera_mtx, dist_params, scaler, model

def lanes_pipeline(img, history, camera_mtx, dist_params, video=False):

    print('Step 1: Undistortion of image')
    undistort = utils.undistort_image(img, camera_mtx, dist_params)

    print('Step 2: Perform perspective transform')
    topview = utils.birdseye(undistort)
    topview_clahe = masks.clahe_filter(topview)
    print('Step 3: Apply image masks')
    yellow_masked = masks.yellow_LAB(topview_clahe)
    white_masked = masks.white_LAB(topview_clahe)
    masked_topview = utils.apply_masks(topview)

    print('Step 4: Find lanes')
    [left_lane, right_lane, history] = utils.find_lanes(masked_topview, history, video)

    print('Step 5: Draw lanes & transform back')
    output = utils.plot_lanes(undistort, left_lane, right_lane)

    return output, history

def vehicles_pipeline(img, history, camera_mtx, dist_params, scaler, model, video=False):

    print('Step 1: Undistortion of image')
    image = utils.undistort_image(img, camera_mtx, dist_params)

    print('Step 2: Find cars')
    detections = find_cars(image, scaler, model)

    print('Step 3: Generate heat map & draw cars')
    [output, history, heatmap] = generate_heatmap(image, detections, history)

    return output, history


from glob import glob

def run(video=False):

    [camera_mtx, dist_params, scaler, model] = initialize()

    lane_history = [[],[]]
    object_history =  Vehicle_Detect()
    # Video=False: Process single, not consecutive images
    if not video:
        file_paths = glob(join(c.IMAGES_TEST_PATH, '*.jpg'))
        file_names = os.listdir(c.IMAGES_TEST_PATH)
        imgs = utils.read_input(file_paths)
        output = []
        for img in imgs:
            result, object_history = vehicles_pipeline(img, object_history, camera_mtx, dist_params, scaler, model)
            # result, lane_history = lanes_pipeline(img, lane_history, camera_mtx, dist_params)
            output.append(result)
        utils.save(output, file_names, None)
    # Video=True: Process videos or consecutive images
    else:
        # file_path = glob(join(c.VIDEO_TEST_PATH, '*.mp4'))
        # frames = utils.read_input(file_path[0], video=True)
        file_path = glob(join(c.FRAMES_TEST_PATH, '*.jpg'))
        file_names = os.listdir(c.FRAMES_TEST_PATH)
        frames = utils.read_input(file_path)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_video = cv2.VideoWriter(c.VIDEO_SAVE_PATH + 'test2_video.mp4', fourcc, 25.0, (1280,720))
        # Run lanes_pipeline in batches
        for offset in range(0, len(frames), c.BATCH_SIZE):
            batch_input = frames[offset:offset+c.BATCH_SIZE]
            batch_output = []
            for frame in batch_input:
                # result, lane_history = lanes_pipeline(frame, lane_history, camera_mtx, dist_params, video)
                result, object_history = vehicles_pipeline(frame, object_history, camera_mtx, dist_params, scaler, model)
                batch_output.append(result)
            utils.save(batch_output, None, output_video, video)

        # Release everything if job is finished
        output_video.release()

# Run code
run(video=True)
