import os
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model
from data_preprocessing import create_data_generator
from constants import MODEL_PATH, IMG_SIZE, BATCH_SIZE, MODEL_CHECKPOINT_PATH

def train_model(train_dir, batch_size=32, epochs=10):
    """Train the model."""
    # Create data generators
    datagen = create_data_generator()
    
    # Get number of classes from directory structure
    num_classes = len(os.listdir(train_dir))
    
    # Create and compile model
    model = create_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(MODEL_CHECKPOINT_PATH, save_best_only=True),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]
    
    # Train model
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save the model with .keras extension
    model.save(MODEL_PATH)
    
    return model, history
