import pathlib
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, losses, models

FILE_DIR = "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/Dataset/crop/"
DATA_PATH = pathlib.Path(FILE_DIR)
IMG_SIZE = (128, 128)  
BATCH_SIZE = 32      
EPOCHS = 2              
datagen = image.ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_generator = datagen.flow_from_directory(
    FILE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',  
    subset='training'
)
val_generator = datagen.flow_from_directory(
    FILE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

nClass = len(train_generator.class_indices) 
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  
    layers.MaxPooling2D(pool_size=(2, 2)),  
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  
    layers.MaxPooling2D(pool_size=(2, 2)),  
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'), 
    layers.MaxPooling2D(pool_size=(2, 2)),  
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dropout(0.5), 
    layers.Dense(nClass, activation='softmax') 
])
model.compile(optimizer='adam', 
              loss=losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
model.save("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/CNN_model.h5")
print("Model saved successfully.")
class_indices = train_generator.class_indices
with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices.json", "w") as f:
    json.dump(class_indices, f)
print("Class indices saved successfully.")
