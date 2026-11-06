import os
import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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
        return None

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    return model


def load_face_detector():

    print("\nLOADING FACE DETECTOR")

    if not os.path.exists(CASCADE_PATH):
        print("Haar Cascade file not found:")
        print(CASCADE_PATH)
        return None

    detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if detector.empty():
        print("Could not load Haar Cascade.")
        return None

    print("Face detector loaded successfully.")

    return detector


def predict_expression(model, face):

    gray_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    resized_face = cv2.resize(
        gray_face,
        IMAGE_SIZE
    )

    normalized_face = (
        resized_face.astype(
            np.float32
        ) / 255.0
    )

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


def main():

    model = load_model()

    if model is None:
        return

    detector = load_face_detector()

    if detector is None:
        return

    print("\nSTARTING WEBCAM")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started successfully.")
    print("Press Q to quit.")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for x, y, w, h in faces:

            face = frame[
                y:y + h,
                x:x + w
            ]

            expression, confidence = (
                predict_expression(
                    model,
                    face
                )
            )

            label = (
                f"{expression} "
                f"{confidence:.1f}%"
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            text_y = y - 10

            if text_y < 25:
                text_y = y + h + 25

            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Real-Time Facial Expression Recognition",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("\nWEBCAM STOPPED")
    print("REAL-TIME RECOGNITION COMPLETED")


if __name__ == "__main__":
    main()