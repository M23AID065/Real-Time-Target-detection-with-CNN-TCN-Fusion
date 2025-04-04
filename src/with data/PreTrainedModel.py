import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tcn import TCN
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# File paths and config
FILE_DIR = "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/Dataset/crop/"
DATA_PATH = pathlib.Path(FILE_DIR)
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 100

# Data preparation
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.2,
    fill_mode='nearest',
    validation_split=0.3
)

train_generator = train_datagen.flow_from_directory(
    FILE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='training'
)
val_generator = train_datagen.flow_from_directory(
    FILE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='validation'
)
nClass = len(train_generator.class_indices)

# Model architecture
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Reshape((-1, 512)),

    TCN(512, return_sequences=True),
    TCN(512, return_sequences=True),
    TCN(256, return_sequences=True),
    TCN(256, return_sequences=True),
    TCN(128, return_sequences=True),
    TCN(128, return_sequences=False),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(nClass, activation='softmax')
])

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
model.summary()
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_scheduler])

# Evaluate and save model
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

model.save("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/CNN_TCN_Deep.h5")
with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices_Fusion.json", 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Model and class indices saved successfully.")

# Plot training history
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
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Generate predictions
y_true = np.concatenate([val_generator[i][1] for i in range(len(val_generator))])
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# ROC Curve
y_true_bin = label_binarize(y_true, classes=np.arange(nClass))
for class_idx in range(nClass):
    fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_pred_probs[:, class_idx])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_generator.class_indices,
            yticklabels=val_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_probs.ravel())
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.show()

# ===============================
# False Positive & False Negative Analysis
# ===============================
class_names = list(val_generator.class_indices.keys())
false_positive_percentages = {}
false_negative_percentages = {}

for cls_idx in range(nClass):
    fp = np.sum((y_pred == cls_idx) & (y_true != cls_idx))
    total_pred_as_class = np.sum(y_pred == cls_idx)
    fp_rate = (fp / total_pred_as_class) * 100 if total_pred_as_class > 0 else 0.0
    false_positive_percentages[class_names[cls_idx]] = round(fp_rate, 2)

    fn = np.sum((y_pred != cls_idx) & (y_true == cls_idx))
    total_actual_class = np.sum(y_true == cls_idx)
    fn_rate = (fn / total_actual_class) * 100 if total_actual_class > 0 else 0.0
    false_negative_percentages[class_names[cls_idx]] = round(fn_rate, 2)

# Plotting FN & FP
labels = list(class_names)
fp_values = [false_positive_percentages[label] for label in labels]
fn_values = [false_negative_percentages[label] for label in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, fp_values, width, label='False Positive %', color='orange')
plt.bar(x + width/2, fn_values, width, label='False Negative %', color='blue')

plt.xlabel('Class Labels')
plt.ylabel('Percentage')
plt.title('False Positive & False Negative Rates per Class')
plt.xticks(x, labels, rotation=45)
plt.legend()

for i, (fp, fn) in enumerate(zip(fp_values, fn_values)):
    plt.text(i - width/2, fp + 0.5, f"{fp}%", ha='center', va='bottom', fontsize=9)
    plt.text(i + width/2, fn + 0.5, f"{fn}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("Model evaluation and visualization completed successfully.")
