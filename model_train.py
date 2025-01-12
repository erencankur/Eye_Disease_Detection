import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"
MODEL_PATH = "eye_disease_detection_model.keras"
IMG_SIZE = 224
CLASSES = ["normal", "glaucoma", "diabetic_retinopathy", "cataract"]
BATCH_SIZE = 32
EPOCHS = 50

def load_data():
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        path = os.path.join(DATASET_PATH, class_name)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error: {img_path} - {str(e)}")
    
    return np.array(images), np.array(labels)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(CLASSES), activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history["accuracy"], label="Eğitim")
    ax1.plot(history.history["val_accuracy"], label="Doğrulama")
    ax1.set_title("Doğruluk")
    ax1.legend()
    
    ax2.plot(history.history["loss"], label="Eğitim")
    ax2.plot(history.history["val_loss"], label="Doğrulama")
    ax2.set_title("Kayip")
    ax2.legend()
    
    plt.show()

def main():
    print("Veri seti yükleniyor...")
    images, labels = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, 
        test_size=0.2,
        random_state=42
    )
    
    print("\nModel oluşturuluyor...")
    model = create_model()
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("\nModel eğitimi başliyor...")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping]
    )
    
    plot_history(history)
    
    _, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest doğruluğu: {test_accuracy:.4f}")
    
    model.save(MODEL_PATH)
    print(f"\nModel kaydedildi: {MODEL_PATH}")

if __name__ == "__main__":
    main()