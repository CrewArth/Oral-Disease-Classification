import cv2
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img / 255.0

def create_data_generator():
    """Create data generator with augmentation."""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
