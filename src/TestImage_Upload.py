import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import json
from tcn import TCN

model = tf.keras.models.load_model(
    "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/CNN_TCN_Optimized.h5",
    custom_objects={'TCN': TCN}
)
with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices_Fusion1.json", 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

def preprocess_image(img_path, img_size=(128, 128)):
    """Load, resize, and normalize the input image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    img_array = cv2.resize(img, img_size)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

def predict_class(img_array):
    """Predict the class of the given image array."""
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class], predictions[0][predicted_class]

image_path = "C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/Dataset/annotated_samples/101a5dc8b7a0104d99ee019a9930f8d9.jpg"  # Change to your image path

try:
    preprocessed_image, original_image = preprocess_image(image_path)
    
    predicted_label, confidence = predict_class(preprocessed_image)
    print(f"Predicted Class: {predicted_label} with confidence {confidence:.2f}")

    cv2.putText(original_image, f"{predicted_label} ({confidence:.2f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image Prediction", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except ValueError as e:
    print(f"Error: {e}")
