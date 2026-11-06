import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "expression_cnn.keras")

X_test = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

print("\nLOADING TRAINED MODEL\n")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nEVALUATING TEST DATA\n")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

print("\nGENERATING PREDICTIONS\n")

y_probability = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_probability, axis=1)

print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX\n")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nPREDICTION DISTRIBUTION\n")

for i, name in enumerate(class_names):
    actual = np.sum(y_test == i)
    predicted = np.sum(y_pred == i)

    print(
        f"{name:8} : "
        f"Actual = {actual:4} | "
        f"Predicted = {predicted:4}"
    )