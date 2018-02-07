from os import makedirs
from os.path import join
import numpy as np

###################################################################
#####################   Define constants   ########################
###################################################################

def get_dir(directory):
     """
     Returns the directory, creating it if it does not exist.
     :param directory: The path for which to get the parent directory.
     :return: The directory.
     """
     try:
          makedirs(directory)  # Recursively creates the current directory
     except OSError:
          pass  # Directory already exists

     return directory

# Define directories
DATA_PATH = '../data/'
CALIBRATION_PATH = join(DATA_PATH, 'camera_cal/')
CALIBRATION_DATA = join(CALIBRATION_PATH, 'calibration_data.p')
IMAGES_TEST_PATH = join(DATA_PATH, 'test_images/')
FRAMES_TEST_PATH = join(DATA_PATH, 'test_frames/')
IMAGES_SAVE_PATH = join(DATA_PATH, 'output_images/')
VIDEO_TEST_PATH = join(DATA_PATH, 'test_video/')
VIDEO_SAVE_PATH = join(DATA_PATH, 'output_video/')

# HOG Classifier
TRAINING_DATA_PATH = join(DATA_PATH, 'object_classifier/training_data/')
MODEL_DIR = get_dir(join(DATA_PATH, 'object_classifier/models'))
MODEL_LOAD_PATH = join(MODEL_DIR, 'model-HLS-15-8-2.pkl')
MODEL_SAVE_PATH = join(MODEL_DIR, 'model-HLS-15-8-2.pkl')

COLORS = [(66, 134, 244), 
          (65, 244, 238),
          (65, 244, 103), 
          (244, 229, 65),
          (66, 134, 244), 
          (244, 157, 65),
          (244, 65, 65), 
          (136, 65, 244),
          (65, 175, 244),
          (241, 65, 244)]

### TODO: Tweak these parameters and see how the results change.
CSPACE = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 15
PPC = 8   # pixel per cell
CPB= 2 # cell per block
HCHNL = 'ALL' # Can be 0, 1, 2, or "ALL"    # hog channel
YSTART = 400
YSTOP = 481

SPATIAL_SIZE = 32
HIST_BINS = 32

# Define image size
IMG_SIZE = (1280, 720)

# Define source and destination points points for the geometric mask
SRC = np.float32(
     [[580, 460],
     [260, 691],
     [1067, 691],
     [705, 460]])

DST = np.float32(
     [[(IMG_SIZE[0] / 4), 0],
     [(IMG_SIZE[0] / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), 0]])

# Define points for the geometric mask
GEO1 = np.float32(
     [[50, 0],
     [(IMG_SIZE[0] / 6), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 10), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 4 / 10), 0]])

GEO2 = np.float32(
     [[(IMG_SIZE[0] * 6 / 10), 0],
     [(IMG_SIZE[0] * 7 / 10), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 5 / 6), IMG_SIZE[1]],
     [(IMG_SIZE[0] - 50), 0]])

# Average lane width in pixels
AVG_LN_WIDTH = 640

# Batch size for video processing
BATCH_SIZE = 100

# meters per pixel in y dimension
MPP_Y = 30. / 720
# meters per pixel in x dimension
MPP_X = 3.7 / AVG_LN_WIDTH

# Camera calibration
NX = 9 # number of inside corners in x
NY = 5 # number of inside corners in y
