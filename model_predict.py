import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

MODEL_PATH = "eye_disease_detection_model.keras"
IMAGE_TO_PREDICT_PATH = "image.jpg"
IMG_SIZE = 224
CLASSES = ["normal", "glaucoma", "diabetic_retinopathy", "cataract"]

def load_prediction_model():
    return load_model(MODEL_PATH)

def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image DNE: {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image DNE: {image_path}")
    
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    processed = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    processed = processed / 255.0
    processed = np.expand_dims(processed, axis=0)
    
    return original, processed

def make_prediction(model, processed_image):
    predictions = model.predict(processed_image)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    
    return {
        "class": CLASSES[class_idx],
        "confidence": confidence
    }

def show_result(image, result):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Prediction: {result['class']}\nConfidence: {result['confidence']:.2%}")
    plt.axis('off')
    plt.show()

def main():
    model = load_prediction_model()
    original_image, processed_image = process_image(IMAGE_TO_PREDICT_PATH)
    result = make_prediction(model, processed_image)
    
    print(f"\nPredicted disease: {result['class']}")
    print(f"Confidence rate: {result['confidence']:.2%}")
    
    show_result(original_image, result)

if __name__ == "__main__":
    main() 