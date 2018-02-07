import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import random

from scipy.ndimage.measurements import label

import constants as c

from hog import *
from utils import convert_color

def random_color():
    """
    This function returns a color from a given set of color values 'c.COLORS'
    :return: RGB color value for random color
    """
    return random.choice(c.COLORS)

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, thick=3):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], random_color(), thick)
    # Return the image copy with boxes drawn
    return imcopy

def slide_window(img, x_start_stop=[None, None], y_start_stop=[c.YSTART, c.YSTOP], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def draw_search_windows(img, scale):

    window_list = slide_window(img, x_start_stop=[None, None], y_start_stop=[c.YSTART, c.YSTART + (c.YSTOP-c.YSTART)*scale], 
                        xy_window=(int(64*scale), int(64*scale)), xy_overlap=(0.5, 0.75))   
    draw_image = draw_boxes(img, window_list)

    return draw_image

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, scaler, model):
    """
    This function searches for vehicles in an image with a pre-trained 'model' 
    and 'scaler'. It defines an image region of interest, search windows
    at different scale and returns all search windows with a positive detection.
    :param img: Image to search.
    :param scaler: Per-column scaler fitted to feature vectors
    :param model: Pre-trained linear SVC model.
    :return: List of search windows with detected vehicles.
    """
    # Set constants
    colorspace = c.CSPACE   # Color space
    orient = c.ORIENT       # Gradient orientation
    pix_per_cell = c.PPC    # Pixels per cell
    cell_per_block = c.CPB  # Cells per block
    hog_channel = c.HCHNL   # Hog channel
    spatial_size = c.SPATIAL_SIZE
    hist_bins = c.HIST_BINS
    # Copy image
    draw_img = np.copy(img)
    search_img = convert_color(img, conv=c.CSPACE)
    # Define scales for search windows
    scales = [1.0, 1.5, 2.0, 2.5, 3.5]
    detections = []
    # Search image with search windows at different scale
    for scale in scales:
        # Get search windows for scale
        window_list = slide_window(img, x_start_stop=[None, None], y_start_stop=[c.YSTART, c.YSTART + (c.YSTOP-c.YSTART)*scale], 
                        xy_window=(int(64*scale), int(64*scale)), xy_overlap=(0.75, 0.75))
        # Search windows for cars
        for window in window_list:
            # Extract window segment from image and convert color space
            segment_tosearch = search_img[window[0][1]:window[1][1],window[0][0]:window[1][0],:]
            # Resize segement if scale > 1
            if scale != 1:
                imshape = segment_tosearch.shape
                segment_tosearch = cv2.resize(segment_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            # Get color channels
            ch1 = segment_tosearch[:,:,0]
            ch2 = segment_tosearch[:,:,1]
            ch3 = segment_tosearch[:,:,2]
            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel() 
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel() 
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel() 
            # Stack HOG features for this patch
            hog_features = np.hstack((hog1, hog2, hog3))
            # Get color features
            # spatial_features = get_color_bin_features(segment_tosearch)
            # hist_features = get_color_hist_features(segment_tosearch)
            # # Stack all features
            # test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = hog_features.reshape(1, -1)
            # Scale features and make a prediction
            test_features = scaler.transform(test_features)    
            test_prediction = model.predict(test_features)
            # If vehicle detected, add to detections  
            if test_prediction == 1:
                detections.append(window)

    return detections

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    boxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
    # Return the image and final rectangles
    return img, boxes

def generate_heatmap(img, detections, history):
    # Generate heatmap for current frame detections
    heatmap_current = np.zeros_like(img[:,:,0])
    heatmap_current_thresh = apply_threshold(add_heat(heatmap_current, detections), 1)
    labels_current = label(heatmap_current_thresh)
    boxes_current = draw_labeled_bboxes(np.copy(img), labels_current)[1]
    # Add car detections (boxes_current) to history
    if len(boxes_current) > 0:
        history.add_detects(boxes_current)
    # Generate heatmap from detections saved in history
    heatmap_img = np.zeros_like(img[:,:,0])
    for detection in history.prev_detects:
        heatmap_img = add_heat(heatmap_img, detection)
    heatmap_thresh = apply_threshold(heatmap_img, 1 + len(history.prev_detects)//2)
    # Get number of detected cars in frame
    labels = label(heatmap_thresh)
    print(labels[1], 'cars found')
    # Draw boxes on detected cars and return image
    draw_img, boxes = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img, history, heatmap_img




