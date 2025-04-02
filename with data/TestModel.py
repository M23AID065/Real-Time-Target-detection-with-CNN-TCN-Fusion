import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, losses, optimizers
from tcn import TCN  
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

FILE_DIR = "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/Dataset/crop/"
DATA_PATH = pathlib.Path(FILE_DIR)
IMG_SIZE = (128, 128)
BATCH_SIZE = 16  
EPOCHS = 50 

datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.3
)

train_generator = datagen.flow_from_directory(
    FILE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='training'
)
val_generator = datagen.flow_from_directory(
    FILE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='validation'
)
nClass = len(train_generator.class_indices)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Reshape((-1, 256)),  
    TCN(256, return_sequences=True),
    TCN(128, return_sequences=True),
    TCN(64, return_sequences=False),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(nClass, activation='softmax')
])

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

model.summary()

history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_scheduler])

model.save("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/CNN_TCN_Optimized.h5")
with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices_Fusion.json", 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Model and class indices saved successfully.")

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend() 
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
plot_training_history(history)

test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
