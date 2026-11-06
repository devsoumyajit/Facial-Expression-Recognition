import os
import sys
import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "expression_cnn.keras"
)

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "src",
    "haarcascade_frontalface_default.xml"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "expression_recognition_result.jpg"
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


def load_face_detector():
    print("\nLOADING FACE DETECTOR")

    if not os.path.exists(CASCADE_PATH):
        print("Haar Cascade file not found:")
        print(CASCADE_PATH)
        sys.exit(1)

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        print("Could not load Haar Cascade.")
        sys.exit(1)

    print("Face detector loaded successfully.")

    return detector


def predict_expression(model, face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    resized_face = cv2.resize(
        gray_face,
        IMAGE_SIZE
    )

    normalized_face = resized_face.astype(
        np.float32
    ) / 255.0

    input_face = normalized_face.reshape(
        1,
        48,
        48,
        1
    )

    predictions = model.predict(
        input_face,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    confidence = (
        predictions[0][predicted_index] * 100
    )

    expression = CLASS_NAMES[
        predicted_index
    ]

    return expression, confidence


def process_image(image_path, model, detector):

    print("\nLOADING IMAGE")

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image:")
        print(image_path)
        sys.exit(1)

    print(f"Original image shape: {image.shape}")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"\nFaces detected: {len(faces)}")

    if len(faces) == 0:
        print("No faces detected.")

    for i, (x, y, w, h) in enumerate(faces):

        print(
            f"\nProcessing Face {i + 1}"
        )

        face = image[
            y:y + h,
            x:x + w
        ]

        expression, confidence = predict_expression(
            model,
            face
        )

        print(
            f"Expression : {expression}"
        )

        print(
            f"Confidence : {confidence:.2f}%"
        )

        label = (
            f"{expression} "
            f"{confidence:.1f}%"
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        text_y = y - 10

        if text_y < 20:
            text_y = y + h + 25

        cv2.putText(
            image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    cv2.imwrite(
        OUTPUT_PATH,
        image
    )

    print("\nResult saved to:")
    print(OUTPUT_PATH)


def main():

    if len(sys.argv) != 2:

        print("Usage:")
        print(
            "python src/expression_recognition.py path_to_image"
        )

        print("\nExample:")
        print(
            "python src/expression_recognition.py test2.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    model = load_model()

    detector = load_face_detector()

    process_image(
        image_path,
        model,
        detector
    )

    print("\nEXPRESSION RECOGNITION COMPLETED")


if __name__ == "__main__":
    main()