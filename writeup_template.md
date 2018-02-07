## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/TrainingData.png
[image2]: ./examples/HOG.png
[image3]: ./examples/scale1_test5.jpg
[image4]: ./examples/scale2.0_test5.jpg
[image5]: ./examples/scale3.5_test5.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

**Loading training data** (`def load_training_data()`: line 25 in `hog.py`)  
The first step was to load all training images and "label" them by seperating `vehicle` from `non-vehicle` images. I used the images from both, KITTI and GTI data base. Some examples of the labeled data:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

**HOG feature extractor** (`def extract_features()`: line 104 in `hog.py`)  
As a next step the classified images were fed into the HOG feature extractor. I decided against the color features and used instead only the hog features. 
I took an empirical approach to find the best setup for the HOG feature extractor. I played with different setups and color spaces and applied them to the test images. In the end, the best setup I found was the following:  

    Color space = "HLS"  
    Orientations = 15  
    Pixels per cell = 8  
    Cells per block = 2  
    HOG channels = 'ALL'  
    Accuracy = 98.54

Here is an example appling the setup above:  

![alt text][image2]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

**Training the SVM model** (`def train_model()`: line 148 in `hog.py`)  
The extracted features are then scaled using a standard scaler, split into training and test data and then fed into a linear support vector machine.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

**Sliding window search** (`def find_cars(img, scaler, model)`: line 83 in `cars.py`)  
For the sliding window search I set the region of interest on the on the lower image area (y-position > 400 pixels). As smallest search window size I used 64x64 (as the training data). These search windows I scaled up to 224x224 size in 5 steps (`scale = [1.0, 1.5, 2.0, 2.5, 3.5]`), focussing with the small windows on the horizontal center of the image and with the bigger search windows also on the lower part of the image (cars in the foreground appear bigger). It showed, that an overlay of 0.75 in each direction, x and y, gave the best results. Below I show the search windows for the scales `1.0`, `2.0` and `3.5`.  

![alt text][image3]
![alt text][image4]
![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

