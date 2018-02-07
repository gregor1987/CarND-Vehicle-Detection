import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog

import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from os.path import join, exists, splitext

import constants as c
from utils import convert_color
import masks

###################################################################
###########################   I / O   #############################
###################################################################

def load_training_data(data_path):
    # Divide up into cars and notcars
    images = glob.glob(join(data_path,'**/*.png'), recursive=True)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    return cars, notcars

def save_model(model, scaler, path=c.MODEL_SAVE_PATH):
    """
    Saves a trained model to file.
    :param model: The trained scikit learn model.
    :param scaler: The scaler used to normalize the data when training.
    :param path: The filepath to which to save the model.
    """
    save_dict = {'model': model, 'scaler':scaler}
    joblib.dump(save_dict, path)


def load_model(path=c.MODEL_SAVE_PATH):
    """
    Loads a trained model from file.
    :param path: The filepath from which to load the model.
    :return: A tuple, (model, scaler).
    """
    save_dict = joblib.load(path)
    print('Model loaded from %s' % path)

    model = save_dict['model']
    scaler = save_dict['scaler']

    return scaler, model

###################################################################
###############   2. Define feature extractor   ###################
###################################################################

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def get_color_bin_features(img, size=(32,32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def get_color_hist_features(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call get_color_bin_features() and get_color_hist_features()
def extract_features(imgs, cspace=c.CSPACE, spatial_size=(32, 32),
                        hist_bins=c.HIST_BINS, orient=c.ORIENT, 
                        pix_per_cell=c.PPC, cell_per_block=c.CPB, hog_channel='ALL',
                        spatial_feat=False, hist_feat=False, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        print(file)
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # image = mpimg.imread(file)
        # image = image*255
        # Convert color space of image
        feature_image = convert_color(image, conv=cspace)  
        # Get color spacial features
        if spatial_feat == True:
            spatial_features = get_color_bin_features(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        # Get color histogram features
        if hist_feat == True:
            # Apply get_color_hist_features()
            hist_features = get_color_hist_features(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        # Get hog features for each channel
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Add hog features to image features
            file_features.append(hog_features)
        # Add image features to overall features list
        features.append(np.concatenate(file_features))

    return features

def train_model(load=True, load_path=c.MODEL_LOAD_PATH, save=True, save_path=c.MODEL_SAVE_PATH):
    """
    Returns a trained model. Trains a new model if load = False or no saved model exists. Otherwise,
    loads and returns the saved model from file.
    :param load: Whether to load a previously-trained model from load_path.
    :param load_path: The path from which to load a trained model if load = True.
    :param save: Whether to save the trained model to save_path.
    :param save_path: The path to which to save the trained model if save - True.
    :return: A model trained to classify car images vs non-car images.
    """
    # If there is a previously trained model and we want to use that, load and return it.
    if load and exists(load_path):
        print('Loading pretrained model...')
        return load_model(path=load_path)
    # Set training constants
    colorspace = c.CSPACE   # Color space
    orient = c.ORIENT       # Gradient orientation
    pix_per_cell = c.PPC    # Pixels per cell
    cell_per_block = c.CPB  # Cells per block
    hog_channel = c.HCHNL   # Hog channel

    # Load training data
    cars, notcars = load_training_data(c.TRAINING_DATA_PATH)
    # Extract features
    t=time.time()
    car_features = extract_features(cars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    print(len(car_features))
    print(len(notcar_features))
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                      
    # Fit a per-column scaler
    scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaler_T = scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaler_T, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    model = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC model...')

    print('Save Model')
    if save:
        save_model(model, scaler, path=save_path)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', model.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return scaler, model

# Train SVC model
# scaler, model = train_model()


###################################################################
######################  Plot example images  ######################
###################################################################

def show_training_data(car_images, noncar_images):

    fig, axs = plt.subplots(8,8, figsize=(15, 15))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    # Step through the list and search for chessboard corners
    for i in np.arange(32):
        img = cv2.imread(car_images[np.random.randint(0,len(car_images))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('Car', fontsize=8)
        axs[i].imshow(img)
    for i in np.arange(32,64):
        img = cv2.imread(noncar_images[np.random.randint(0,len(noncar_images))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('No Car', fontsize=8)
        axs[i].imshow(img)
    plt.savefig('TrainingData.png')
    plt.show()


# Show training data
cars, notcars = load_training_data(c.TRAINING_DATA_PATH)
# show_training_data(cars, notcars)

    # # Display the image
    # f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    # ax1.imshow(heatmap, cmap='hot')
    # ax2.imshow(output)
    # plt.savefig('heat.png')
    # plt.show()

    # plt.figure(figsize=(10,5))
    # plt.imshow(output)
    # plt.show()
    # history = Vehicle_Detect()
