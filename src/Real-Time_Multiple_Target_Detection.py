import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import json
from tcn import TCN

model = tf.keras.models.load_model(r"C:\Users\lohit ramaraju\OneDrive\Desktop\IITJ\Main Project\saved_models\CNN_TCN_Optimized.h5",
    custom_objects={'TCN': TCN}
)

with open("C:/Users/lohit ramaraju/OneDrive/Desktop/IITJ/Main Project/saved_models/class_indices_Fusion1.json", 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

def preprocess_image(img, img_size=(128, 128)):
    """Resize and normalize the input image."""
    img_array = cv2.resize(img, img_size)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_class(img_array):
    """Predict the class of the given image array."""
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class], predictions[0][predicted_class]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    preprocessed_image = preprocess_image(frame)

    predicted_label, confidence = predict_class(preprocessed_image)
    print(f"Predicted Class: {predicted_label} with confidence {confidence:.2f}")
    cv2.putText(frame, f"{predicted_label} ({confidence:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
