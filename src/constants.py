import os

# Directory paths
DATASET_DIR = 'Dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'TRAIN')
TEST_DIR = os.path.join(DATASET_DIR, 'TEST')

# Model settings
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
MODEL_CHECKPOINT_PATH = 'models/best_model.keras'

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
