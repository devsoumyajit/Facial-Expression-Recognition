import os
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from database import (
    create_database,
    start_session,
    insert_prediction,
    end_session
)


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

WINDOW_NAME = (
    "Real-Time Facial Expression Recognition"
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

SMOOTHING_FRAMES = 7

# Save one prediction every 3 seconds
DATABASE_LOG_INTERVAL = 3.0


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("LOADING TRAINED MODEL")

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "Model file not found:"
        )

        print(
            MODEL_PATH
        )

        return None

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

def load_face_detector():

    print(
        "\nLOADING FACE DETECTOR"
    )

    if not os.path.exists(
        CASCADE_PATH
    ):

        print(
            "Haar Cascade file not found:"
        )

        print(
            CASCADE_PATH
        )

        return None

    detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if detector.empty():

        print(
            "Could not load Haar Cascade."
        )

        return None

    print(
        "Face detector loaded successfully."
    )

    return detector


# ============================================================
# PREDICT EXPRESSION
# ============================================================

def predict_expression(
    model,
    face
):

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

    input_face = (
        normalized_face.reshape(
            1,
            48,
            48,
            1
        )
    )

    predictions = model.predict(
        input_face,
        verbose=0
    )

    probabilities = (
        predictions[0]
    )

    predicted_index = np.argmax(
        probabilities
    )

    expression = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = (
        probabilities[
            predicted_index
        ] * 100
    )

    return (
        expression,
        confidence,
        probabilities
    )


# ============================================================
# RESIZE FRAME TO FULLSCREEN
# ============================================================

def resize_to_fill_screen(
    frame
):

    (
        window_x,
        window_y,
        screen_width,
        screen_height
    ) = cv2.getWindowImageRect(
        WINDOW_NAME
    )

    if (
        screen_width <= 0
        or screen_height <= 0
    ):

        return frame

    frame_height, frame_width = (
        frame.shape[:2]
    )

    frame_ratio = (
        frame_width
        / frame_height
    )

    screen_ratio = (
        screen_width
        / screen_height
    )

    if frame_ratio > screen_ratio:

        new_height = (
            screen_height
        )

        new_width = int(
            new_height
            * frame_ratio
        )

    else:

        new_width = (
            screen_width
        )

        new_height = int(
            new_width
            / frame_ratio
        )

    resized_frame = cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    crop_x = (
        new_width
        - screen_width
    ) // 2

    crop_y = (
        new_height
        - screen_height
    ) // 2

    cropped_frame = (
        resized_frame[
            crop_y:
            crop_y
            + screen_height,

            crop_x:
            crop_x
            + screen_width
        ]
    )

    return cropped_frame


# ============================================================
# STABLE PREDICTION
# ============================================================

def get_stable_prediction(
    prediction_history
):

    if len(
        prediction_history
    ) == 0:

        return (
            None,
            0.0
        )

    predicted_indices = [
        item[0]
        for item in prediction_history
    ]

    counts = np.bincount(
        predicted_indices,
        minlength=len(
            CLASS_NAMES
        )
    )

    stable_index = (
        np.argmax(
            counts
        )
    )

    stable_probabilities = []

    for item in (
        prediction_history
    ):

        probabilities = (
            item[1]
        )

        stable_probabilities.append(
            probabilities[
                stable_index
            ]
        )

    stable_confidence = (
        np.mean(
            stable_probabilities
        ) * 100
    )

    return (
        stable_index,
        stable_confidence
    )


# ============================================================
# CREATE SESSION ID
# ============================================================

def create_session_id():

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    unique_id = (
        uuid.uuid4().hex[:6]
    )

    return (
        f"{timestamp}_{unique_id}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CREATE DATABASE
    # --------------------------------------------------------

    create_database()

    # --------------------------------------------------------
    # CREATE SESSION
    # --------------------------------------------------------

    session_id = (
        create_session_id()
    )

    print(
        f"\nSESSION ID: "
        f"{session_id}"
    )

    start_session(
        session_id
    )

    print(
        "Session started successfully."
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    if model is None:

        return

    # --------------------------------------------------------
    # LOAD FACE DETECTOR
    # --------------------------------------------------------

    detector = (
        load_face_detector()
    )

    if detector is None:

        return

    # --------------------------------------------------------
    # CONNECT TO CAMERA
    # --------------------------------------------------------

    print(
        "\nCONNECTING TO USB DROIDCAM"
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_MSMF
    )

    if not camera.isOpened():

        print(
            "Could not open USB camera."
        )

        return

    print(
        "USB DroidCam connected successfully."
    )

    camera_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    camera_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Camera resolution: "
        f"{camera_width} x "
        f"{camera_height}"
    )

    print(
        f"Smoothing frames: "
        f"{SMOOTHING_FRAMES}"
    )

    print(
        f"Database log interval: "
        f"{DATABASE_LOG_INTERVAL} seconds"
    )

    print(
        "\nSTARTING REAL-TIME RECOGNITION"
    )

    print(
        "Press Q to quit."
    )

    # --------------------------------------------------------
    # FULLSCREEN WINDOW
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # --------------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------------

    prediction_history = (
        deque(
            maxlen=SMOOTHING_FRAMES
        )
    )

    # --------------------------------------------------------
    # DATABASE TIMER
    # --------------------------------------------------------

    last_database_log_time = 0

    # --------------------------------------------------------
    # MAIN CAMERA LOOP
    # --------------------------------------------------------

    while True:

        ret, frame = (
            camera.read()
        )

        if not ret:

            print(
                "Could not read "
                "camera frame."
            )

            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = (
            detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
        )

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if len(faces) == 0:

            prediction_history.clear()

        # ----------------------------------------------------
        # PROCESS EACH FACE
        # ----------------------------------------------------

        for (
            x,
            y,
            w,
            h
        ) in faces:

            face = frame[
                y:y + h,
                x:x + w
            ]

            (
                expression,
                confidence,
                probabilities
            ) = predict_expression(
                model,
                face
            )

            predicted_index = (
                CLASS_NAMES.index(
                    expression
                )
            )

            prediction_history.append(
                (
                    predicted_index,
                    probabilities
                )
            )

            (
                stable_index,
                stable_confidence
            ) = get_stable_prediction(
                prediction_history
            )

            stable_expression = (
                CLASS_NAMES[
                    stable_index
                ]
            )

            # ------------------------------------------------
            # DISPLAY LABEL
            # ------------------------------------------------

            label = (
                f"{stable_expression} "
                f"{stable_confidence:.1f}%"
            )

            # ------------------------------------------------
            # FACE RECTANGLE
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # LABEL POSITION
            # ------------------------------------------------

            text_y = (
                y - 10
            )

            if text_y < 25:

                text_y = (
                    y
                    + h
                    + 25
                )

            # ------------------------------------------------
            # EXPRESSION TEXT
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # STABILITY
            # ------------------------------------------------

            stable_count = sum(
                1
                for item
                in prediction_history
                if item[0]
                == stable_index
            )

            stability_text = (
                f"Stable: "
                f"{stable_count}/"
                f"{SMOOTHING_FRAMES}"
            )

            cv2.putText(
                frame,
                stability_text,
                (
                    x,
                    text_y + 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            # ------------------------------------------------
            # SAVE TO DATABASE
            # ------------------------------------------------

            current_time = (
                cv2.getTickCount()
                / cv2.getTickFrequency()
            )

            if (
                current_time
                - last_database_log_time
                >= DATABASE_LOG_INTERVAL
            ):

                insert_prediction(
                    session_id=(
                        session_id
                    ),
                    expression=(
                        stable_expression
                    ),
                    confidence=(
                        stable_confidence
                    ),
                    faces_detected=(
                        len(faces)
                    )
                )

                last_database_log_time = (
                    current_time
                )

                print(
                    "\nDATABASE LOGGED"
                )

                print(
                    f"Expression : "
                    f"{stable_expression}"
                )

                print(
                    f"Confidence : "
                    f"{stable_confidence:.2f}%"
                )

                print(
                    f"Faces      : "
                    f"{len(faces)}"
                )

        # ----------------------------------------------------
        # DISPLAY FRAME
        # ----------------------------------------------------

        display_frame = (
            resize_to_fill_screen(
                frame
            )
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            break

    # --------------------------------------------------------
    # STOP CAMERA
    # --------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # END SESSION
    # --------------------------------------------------------

    end_session(
        session_id
    )

    print(
        "\nUSB DROIDCAM STOPPED"
    )

    print(
        "REAL-TIME RECOGNITION "
        "COMPLETED"
    )

    print(
        f"\nSession ID: "
        f"{session_id}"
    )

    print(
        "Session summary saved "
        "to database."
    )


if __name__ == "__main__":

    main()