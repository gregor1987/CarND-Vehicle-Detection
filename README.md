## Writeup

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
[image6]: ./examples/boxes_test6.jpg
[image7]: ./examples/heat7.png
[image8]: ./examples/output_test5.jpg
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

The image below shows the overall result of the sliding window search, adding up the detection from different search window sizes. One the left side we can see a false positive detection. This false positive, however, can be easily filtered by considering car detections only valid if minimum number of search windows detected a car in the same area of the image.  

![alt text][image6]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/HURFcv4pae8)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

**Heat map and plausibilization**(lines 145 through 197 in `cars.py`)
Having multiple detections, a heat map can be applied (left side of the image below). For plausibilzation of the detection I used a combination of 

1. `Thresholding of the heatmap`  
   Only areas of the image, which had more than one detection, were selected valid.   
2. Applying `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap
3. `History evaluation`  
    It is evaluated if a car had been already detected in the labeled area of the image in a previous frame. If true, the detection is considered valid.

![alt text][image7]
![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My implementation is far from being real-time capable. Instead of using the somewhat classical HOG feature extractor approach, one could go for more advanced techniques using deep learning approaches such as `YOLO`, which has been already impressively applied to the project video by a Udacity fellow here: 
(https://medium.com/@ksakmann/vehicle-detection-and-tracking-using-hog-features-svm-vs-yolo-73e1ccb35866)
