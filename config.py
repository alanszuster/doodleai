"""
Configuration file for AI Drawing Classifier project.
"""

import os

# Project structure
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
DOCKER_DIR = os.path.join(PROJECT_ROOT, 'docker')

# Dataset configuration
QUICKDRAW_DIR = os.path.join(DATASET_DIR, 'quickdraw')
PROCESSED_DIR = os.path.join(DATASET_DIR, 'processed')
QUICKDRAW_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

# Model configuration
MODEL_PATH = os.path.join(MODEL_DIR, 'drawing_model.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.json')

# Training configuration
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MAX_SAMPLES_PER_CLASS = 15000

# Data split configuration
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42

# Classes to train on (verified as available in Quick Draw! dataset)
CLASSES = [
    # Animals
    'bear', 'bee', 'butterfly', 'cat', 'cow', 'crab', 'deer', 'dog',
    'dolphin', 'duck', 'elephant', 'fish', 'flamingo', 'frog', 'giraffe',
    'hedgehog', 'horse', 'kangaroo', 'lion', 'monkey', 'octopus', 'owl',
    'panda', 'penguin', 'pig', 'rabbit', 'shark', 'sheep', 'snake',
    'spider', 'tiger', 'whale', 'zebra',
    # Food
    'apple', 'banana', 'birthday cake', 'bread', 'carrot', 'cookie',
    'donut', 'grapes', 'hamburger', 'hot dog', 'ice cream', 'lemon',
    'mushroom', 'pear', 'pineapple', 'pizza', 'strawberry', 'watermelon',
    # Vehicles
    'airplane', 'bicycle', 'bus', 'car', 'firetruck', 'helicopter',
    'motorbike', 'rocket', 'sailboat', 'submarine', 'train', 'truck',
    # Objects
    'backpack', 'book', 'camera', 'chair', 'clock', 'computer', 'cup',
    'drums', 'fork', 'guitar', 'hammer', 'hat', 'key', 'knife', 'lamp',
    'microphone', 'pencil', 'piano', 'scissors', 'shoe', 'sword', 'umbrella',
    # Nature
    'cloud', 'fire', 'flower', 'leaf', 'lightning', 'moon', 'mountain',
    'rainbow', 'snowflake', 'star', 'sun', 'tree',
    # Buildings
    'bridge', 'castle', 'door', 'fence', 'house', 'lighthouse', 'windmill',
    # Body
    'ear', 'eye', 'face', 'hand', 'nose', 'tooth',
    # Misc
    'circle', 'crown', 'diamond', 'heart', 'hot air balloon', 'lollipop',
    'skull', 'stop sign', 'tornado', 'trophy',
]
