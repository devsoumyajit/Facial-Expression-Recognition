import os
import sys
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "expression_cnn.keras"
)

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

IMAGE_SIZE = (48, 48)


def load_model():
    print("LOADING TRAINED MODEL")

    if not os.path.exists(MODEL_PATH):
        print("Model file not found:")
        print(MODEL_PATH)
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")
    return model


def preprocess_image(image_path):
    print("\nLOADING IMAGE")

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image:")
        print(image_path)
        sys.exit(1)

    print(f"Original image shape: {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, IMAGE_SIZE)

    normalized = resized.astype(np.float32) / 255.0

    input_image = normalized.reshape(1, 48, 48, 1)

    print(f"Model input shape: {input_image.shape}")

    return input_image


def predict_expression(model, input_image):
    print("\nGENERATING PREDICTION")

    predictions = model.predict(input_image, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    expression = CLASS_NAMES[predicted_index]

    print(f"Predicted Expression : {expression}")
    print(f"Confidence           : {confidence:.2f}%")

    print("\nALL PREDICTION PROBABILITIES")

    for i, class_name in enumerate(CLASS_NAMES):
        probability = predictions[0][i] * 100
        print(f"{class_name:<10}: {probability:.2f}%")

    return expression, confidence


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python src/predict.py path_to_image")

        print("\nExample:")
        print("python src/predict.py test.jpg")

        sys.exit(1)

    image_path = sys.argv[1]

    model = load_model()

    input_image = preprocess_image(image_path)

    predict_expression(model, input_image)

    print("\nPREDICTION COMPLETED")


if __name__ == "__main__":
    main()