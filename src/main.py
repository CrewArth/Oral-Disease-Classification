import os
from train import train_model
from evaluate import evaluate_model, plot_training_history
from data_preprocessing import create_data_generator
from constants import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    """
    Main entry point of the program.
    Trains and evaluates an image classification model.
    Requires dataset to be organized in Dataset/TRAIN and Dataset/TEST folders.
    """
    print("Starting image classification training pipeline...")
    
    # Train model
    model, history = train_model(TRAIN_DIR)
    
    # Create test generator
    datagen = create_data_generator()
    test_generator = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    report = evaluate_model(model, test_generator)
    print("\nClassification Report:")
    print(report)
    
    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
