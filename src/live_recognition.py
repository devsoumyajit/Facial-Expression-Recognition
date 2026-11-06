import os
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

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
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

# ------------------------------------------------------------
# Performance settings
# ------------------------------------------------------------

# AI processing will happen only once every N frames.
# The camera viewer itself will continue showing every frame.
PROCESS_EVERY_N_FRAMES = 5

# SQLite logging interval
DATABASE_LOG_INTERVAL = 3.0

# Small delay to prevent excessive CPU usage
FRAME_DELAY = 0.01


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Live Recognition",
    page_icon="📷",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        return None

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    if not os.path.exists(
        CASCADE_PATH
    ):

        return None

    detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if detector.empty():

        return None

    return detector


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

    predictions = (
        model.predict(
            input_face,
            verbose=0
        )
    )

    probabilities = (
        predictions[0]
    )

    predicted_index = (
        np.argmax(
            probabilities
        )
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
        confidence
    )


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(
    frame,
    model,
    detector
):

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

    results = []

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
            confidence
        ) = predict_expression(
            model,
            face
        )

        results.append(
            (
                expression,
                confidence,
                x,
                y,
                w,
                h
            )
        )

        # ----------------------------------------------------
        # FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # EXPRESSION LABEL
        # ----------------------------------------------------

        label = (
            f"{expression} "
            f"{confidence:.1f}%"
        )

        text_y = (
            y - 10
        )

        if text_y < 25:

            text_y = (
                y + h + 25
            )

        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return (
        frame,
        results
    )


# ============================================================
# DRAW PREVIOUS RESULTS
# ============================================================

def draw_previous_results(
    frame,
    results
):

    for (
        expression,
        confidence,
        x,
        y,
        w,
        h
    ) in results:

        # ----------------------------------------------------
        # FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"{expression} "
            f"{confidence:.1f}%"
        )

        text_y = (
            y - 10
        )

        if text_y < 25:

            text_y = (
                y + h + 25
            )

        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return frame


# ============================================================
# TITLE
# ============================================================

st.title(
    "Live Facial Expression Recognition"
)

st.write(
    "Real-time facial expression recognition "
    "using the trained CNN model."
)


# ============================================================
# LOAD COMPONENTS
# ============================================================

model = load_model()

detector = load_face_detector()


# ============================================================
# CHECK MODEL
# ============================================================

if model is None:

    st.error(
        "Trained model not found."
    )

    st.code(
        MODEL_PATH
    )

    st.stop()


# ============================================================
# CHECK DETECTOR
# ============================================================

if detector is None:

    st.error(
        "Haar Cascade file could not be loaded."
    )

    st.code(
        CASCADE_PATH
    )

    st.stop()


# ============================================================
# DATABASE
# ============================================================

create_database()


# ============================================================
# CAMERA CONTROLS
# ============================================================

st.markdown(
    "### Camera Controls"
)

col1, col2 = (
    st.columns(2)
)

with col1:

    start_camera = st.button(
        "Start Camera",
        type="primary"
    )

with col2:

    stop_camera = st.button(
        "Stop Camera"
    )


# ============================================================
# CAMERA STATE
# ============================================================

if (
    "camera_running"
    not in st.session_state
):

    st.session_state[
        "camera_running"
    ] = False


if (
    "session_id"
    not in st.session_state
):

    st.session_state[
        "session_id"
    ] = None


if (
    "last_database_log_time"
    not in st.session_state
):

    st.session_state[
        "last_database_log_time"
    ] = 0


# ============================================================
# START CAMERA
# ============================================================

if start_camera:

    if not st.session_state[
        "camera_running"
    ]:

        session_id = (
            create_session_id()
        )

        start_session(
            session_id
        )

        st.session_state[
            "session_id"
        ] = session_id

        st.session_state[
            "camera_running"
        ] = True

        st.session_state[
            "last_database_log_time"
        ] = 0


# ============================================================
# STOP CAMERA
# ============================================================

if stop_camera:

    if st.session_state[
        "camera_running"
    ]:

        session_id = (
            st.session_state[
                "session_id"
            ]
        )

        if session_id is not None:

            end_session(
                session_id
            )

        st.session_state[
            "camera_running"
        ] = False

        st.session_state[
            "session_id"
        ] = None


# ============================================================
# STATUS
# ============================================================

status_placeholder = (
    st.empty()
)

camera_placeholder = (
    st.empty()
)

session_placeholder = (
    st.empty()
)


# ============================================================
# CAMERA RUNNING
# ============================================================

if st.session_state[
    "camera_running"
]:

    session_placeholder.info(
        f"Session ID: "
        f"{st.session_state['session_id']}"
    )

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_MSMF
    )

    if not camera.isOpened():

        st.error(
            "Could not open camera."
        )

        st.session_state[
            "camera_running"
        ] = False

        st.stop()

    status_placeholder.success(
        "Camera connected successfully."
    )

    # --------------------------------------------------------
    # PERFORMANCE VARIABLES
    # --------------------------------------------------------

    frame_count = 0

    last_results = []

    # --------------------------------------------------------
    # CAMERA LOOP
    # --------------------------------------------------------

    while (
        st.session_state[
            "camera_running"
        ]
    ):

        # ====================================================
        # READ FRAME
        # ====================================================

        ret, frame = (
            camera.read()
        )

        if not ret:

            status_placeholder.error(
                "Could not read camera frame."
            )

            break

        frame_count += 1

        # ====================================================
        # AI PROCESSING
        # ====================================================

        if (
            frame_count
            %
            PROCESS_EVERY_N_FRAMES
            == 0
        ):

            (
                frame,
                results
            ) = process_frame(
                frame,
                model,
                detector
            )

            # Save latest AI result
            last_results = (
                results
            )

        else:

            # =================================================
            # DRAW LAST RESULT
            # =================================================

            frame = (
                draw_previous_results(
                    frame,
                    last_results
                )
            )

            results = (
                last_results
            )

        # ====================================================
        # DATABASE LOGGING
        # ====================================================

        current_time = (
            time.time()
        )

        if (
            len(results) > 0
            and
            (
                current_time
                -
                st.session_state[
                    "last_database_log_time"
                ]
                >=
                DATABASE_LOG_INTERVAL
            )
        ):

            expression = (
                results[0][0]
            )

            confidence = (
                results[0][1]
            )

            insert_prediction(
                session_id=(
                    st.session_state[
                        "session_id"
                    ]
                ),
                expression=(
                    expression
                ),
                confidence=(
                    confidence
                ),
                faces_detected=(
                    len(results)
                )
            )

            st.session_state[
                "last_database_log_time"
            ] = (
                current_time
            )

        # ====================================================
        # RGB CONVERSION
        # ====================================================

        frame_rgb = (
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        # ====================================================
        # DISPLAY CAMERA
        # ====================================================

        camera_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        # ====================================================
        # SMALL DELAY
        # ====================================================

        time.sleep(
            FRAME_DELAY
        )

    # ========================================================
    # RELEASE CAMERA
    # ========================================================

    camera.release()

    status_placeholder.info(
        "Camera stopped."
    )

else:

    status_placeholder.info(
        "Camera is currently stopped."
    )

    st.info(
        "Click 'Start Camera' to begin "
        "real-time facial expression recognition."
    )