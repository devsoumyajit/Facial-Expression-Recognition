import os
import base64
import sqlite3
import uuid
import time

import cv2
import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string
)

from datetime import datetime


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# PROJECT PATHS
# ==========================================================

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


DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)


DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "expression_recognition.db"
)


# ==========================================================
# SETTINGS
# ==========================================================

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


# ----------------------------------------------------------
# DATABASE LOGGING
# ----------------------------------------------------------

DATABASE_LOG_INTERVAL = 3.0


# ----------------------------------------------------------
# CAMERA / PREDICTION SETTINGS
# ----------------------------------------------------------

# Browser sends a frame every 700 ms.
# This is faster than the previous 1000 ms,
# but still much safer for Render than 500 ms.

PREDICTION_INTERVAL_MS = 700


# JPEG quality sent from browser to server.

JPEG_QUALITY = 0.65


# ----------------------------------------------------------
# IMAGE PROCESSING LIMIT
# ----------------------------------------------------------

# Maximum width of the image sent to OpenCV.
#
# This reduces CPU usage on Render.
#
# The browser camera can still display normally.

MAX_PROCESSING_WIDTH = 640


# ==========================================================
# TENSORFLOW CPU SETTINGS
# ==========================================================

# Render free instances normally do not have a GPU.
#
# Limiting TensorFlow threads helps prevent excessive
# CPU usage and worker instability.

try:

    tf.config.threading.set_intra_op_parallelism_threads(
        1
    )

    tf.config.threading.set_inter_op_parallelism_threads(
        1
    )

except Exception:

    pass


# ==========================================================
# LOAD CNN MODEL
# ==========================================================

print("=" * 60)
print("LOADING CNN MODEL")
print("=" * 60)


if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH
)


print(
    "CNN model loaded successfully."
)


# ==========================================================
# LOAD HAAR FACE DETECTOR
# ==========================================================

print()
print("=" * 60)
print("LOADING FACE DETECTOR")
print("=" * 60)


if not os.path.exists(
    CASCADE_PATH
):

    raise FileNotFoundError(
        f"Haar Cascade not found:\n{CASCADE_PATH}"
    )


face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


if face_detector.empty():

    raise RuntimeError(
        "Failed to load Haar Cascade."
    )


print(
    "Haar Cascade loaded successfully."
)


# ==========================================================
# SESSION STATE
# ==========================================================

active_sessions = {}

last_database_log_time = {}


# ==========================================================
# DATABASE
# ==========================================================

def create_database():

    os.makedirs(
        DATABASE_DIR,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    # ------------------------------------------------------
    # PREDICTIONS TABLE
    # ------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id TEXT,

            timestamp TEXT,

            expression TEXT,

            confidence REAL,

            faces_detected INTEGER

        )
    """)


    # ------------------------------------------------------
    # SESSIONS TABLE
    # ------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id TEXT UNIQUE,

            start_time TEXT,

            end_time TEXT,

            duration_seconds REAL,

            total_predictions INTEGER,

            most_detected_expression TEXT,

            average_confidence REAL,

            average_faces_detected REAL

        )
    """)


    connection.commit()
    connection.close()


# ==========================================================
# START SESSION
# ==========================================================

def start_session():

    session_id = (
        "WEB_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:6].upper()
    )


    start_time = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    cursor.execute("""
        INSERT INTO sessions (
            session_id,
            start_time
        )
        VALUES (?, ?)
    """, (
        session_id,
        start_time
    ))


    connection.commit()
    connection.close()


    active_sessions[
        session_id
    ] = {
        "start_time":
            time.time()
    }


    last_database_log_time[
        session_id
    ] = 0


    return session_id


# ==========================================================
# INSERT PREDICTION
# ==========================================================

def insert_prediction(
    session_id,
    expression,
    confidence,
    faces_detected
):

    timestamp = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    cursor.execute("""
        INSERT INTO predictions (
            session_id,
            timestamp,
            expression,
            confidence,
            faces_detected
        )
        VALUES (?, ?, ?, ?, ?)
    """, (

        session_id,

        timestamp,

        expression,

        float(
            confidence
        ),

        int(
            faces_detected
        )

    ))


    connection.commit()
    connection.close()


# ==========================================================
# END SESSION
# ==========================================================

def end_session(
    session_id
):

    if (
        session_id
        not in active_sessions
    ):

        return {
            "success":
                False,

            "message":
                "Session not found."
        }


    end_time = time.time()


    start_time = (
        active_sessions[
            session_id
        ]["start_time"]
    )


    duration = (
        end_time
        -
        start_time
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    # ------------------------------------------------------
    # GET SESSION PREDICTIONS
    # ------------------------------------------------------

    cursor.execute("""
        SELECT
            expression,
            confidence,
            faces_detected

        FROM predictions

        WHERE session_id = ?
    """, (
        session_id,
    ))


    predictions = (
        cursor.fetchall()
    )


    total_predictions = (
        len(
            predictions
        )
    )


    # ------------------------------------------------------
    # CALCULATE ANALYTICS
    # ------------------------------------------------------

    if (
        total_predictions
        > 0
    ):

        expressions = [
            row[0]
            for row in predictions
        ]


        confidences = [
            float(row[1])
            for row in predictions
        ]


        faces = [
            int(row[2])
            for row in predictions
        ]


        from collections import Counter


        expression_counts = (
            Counter(
                expressions
            )
        )


        most_detected_expression = (
            expression_counts
            .most_common(1)[0][0]
        )


        average_confidence = (
            sum(
                confidences
            )
            /
            total_predictions
        )


        average_faces = (
            sum(
                faces
            )
            /
            total_predictions
        )

    else:

        most_detected_expression = (
            "None"
        )

        average_confidence = 0

        average_faces = 0


    end_time_string = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    # ------------------------------------------------------
    # UPDATE SESSION
    # ------------------------------------------------------

    cursor.execute("""
        UPDATE sessions

        SET
            end_time = ?,
            duration_seconds = ?,
            total_predictions = ?,
            most_detected_expression = ?,
            average_confidence = ?,
            average_faces_detected = ?

        WHERE session_id = ?
    """, (

        end_time_string,

        duration,

        total_predictions,

        most_detected_expression,

        average_confidence,

        average_faces,

        session_id

    ))


    connection.commit()
    connection.close()


    # ------------------------------------------------------
    # REMOVE SESSION FROM MEMORY
    # ------------------------------------------------------

    del active_sessions[
        session_id
    ]


    if (
        session_id
        in last_database_log_time
    ):

        del last_database_log_time[
            session_id
        ]


    return {

        "success":
            True,

        "session_id":
            session_id,

        "duration_seconds":
            duration,

        "total_predictions":
            total_predictions,

        "most_detected_expression":
            most_detected_expression,

        "average_confidence":
            average_confidence,

        "average_faces_detected":
            average_faces

    }


# ==========================================================
# FACE PREDICTION
# ==========================================================

def predict_expression(
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
        )
        /
        255.0
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


    predicted_index = (
        np.argmax(
            predictions[0]
        )
    )


    confidence = (
        predictions[0][
            predicted_index
        ]
        *
        100
    )


    expression = (
        CLASS_NAMES[
            predicted_index
        ]
    )


    return (
        expression,
        float(
            confidence
        )
    )


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template_string(
        HTML_PAGE
    )


# ==========================================================
# START SESSION API
# ==========================================================

@app.route(
    "/start_session",
    methods=["POST"]
)
def start_session_api():

    try:

        session_id = (
            start_session()
        )


        return jsonify({

            "success":
                True,

            "session_id":
                session_id

        })


    except Exception as e:

        print(
            "Start session error:",
            str(e)
        )


        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# ==========================================================
# END SESSION API
# ==========================================================

@app.route(
    "/end_session",
    methods=["POST"]
)
def end_session_api():

    try:

        data = (
            request.get_json()
            or {}
        )


        session_id = (
            data.get(
                "session_id"
            )
        )


        if not session_id:

            return jsonify({

                "success":
                    False,

                "message":
                    "Session ID is required."

            })


        result = (
            end_session(
                session_id
            )
        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "End session error:",
            str(e)
        )


        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# ==========================================================
# PREDICTION API
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = (
            request.get_json()
            or {}
        )


        image_data = (
            data.get(
                "image"
            )
        )


        session_id = (
            data.get(
                "session_id"
            )
        )


        if not image_data:

            return jsonify({

                "success":
                    False,

                "message":
                    "Image data missing."

            })


        # --------------------------------------------------
        # DECODE BASE64 IMAGE
        # --------------------------------------------------

        try:

            encoded_image = (
                image_data.split(
                    ",",
                    1
                )[1]
            )

        except IndexError:

            encoded_image = (
                image_data
            )


        image_bytes = (
            base64.b64decode(
                encoded_image
            )
        )


        image_array = (
            np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )
        )


        image = (
            cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )
        )


        if image is None:

            return jsonify({

                "success":
                    False,

                "message":
                    "Could not decode image."

            })


        # ==================================================
        # RESIZE FOR RENDER PROCESSING
        # ==================================================

        original_height, original_width = (
            image.shape[:2]
        )


        if (
            original_width
            >
            MAX_PROCESSING_WIDTH
        ):

            scale = (
                MAX_PROCESSING_WIDTH
                /
                original_width
            )


            new_width = (
                MAX_PROCESSING_WIDTH
            )


            new_height = int(
                original_height
                *
                scale
            )


            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height
                ),
                interpolation=
                    cv2.INTER_AREA
            )


        # ==================================================
        # FACE DETECTION
        # ==================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        faces = (
            face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )
        )


        faces_data = []


        # ==================================================
        # PROCESS EVERY FACE
        # ==================================================

        for (
            x,
            y,
            w,
            h
        ) in faces:

            face = image[
                y:y + h,
                x:x + w
            ]


            expression, confidence = (
                predict_expression(
                    face
                )
            )


            faces_data.append({

                "x":
                    int(x),

                "y":
                    int(y),

                "width":
                    int(w),

                "height":
                    int(h),

                "expression":
                    expression,

                "confidence":
                    round(
                        confidence,
                        2
                    )

            })


        # ==================================================
        # DATABASE LOGGING
        # ==================================================

        if (
            session_id
            and
            session_id
            in active_sessions
            and
            len(
                faces_data
            )
            > 0
        ):

            current_time = (
                time.time()
            )


            last_log_time = (
                last_database_log_time.get(
                    session_id,
                    0
                )
            )


            if (
                current_time
                -
                last_log_time
                >=
                DATABASE_LOG_INTERVAL
            ):

                faces_count = (
                    len(
                        faces_data
                    )
                )


                # ------------------------------------------
                # Store one record per face
                # ------------------------------------------

                for face_data in (
                    faces_data
                ):

                    insert_prediction(

                        session_id,

                        face_data[
                            "expression"
                        ],

                        face_data[
                            "confidence"
                        ],

                        faces_count

                    )


                last_database_log_time[
                    session_id
                ] = current_time


        # ==================================================
        # RETURN RESULT
        # ==================================================

        return jsonify({

            "success":
                True,

            "faces":
                len(
                    faces_data
                ),

            "faces_data":
                faces_data

        })


    except Exception as e:

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)
        print(
            str(e)
        )


        return jsonify({

            "success":
                False,

            "message":
                str(e)

        }), 500


# ==========================================================
# HTML + JAVASCRIPT
# ==========================================================

HTML_PAGE = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0">

<title>
Facial Expression Recognition
</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        #f4f6f8;

    color:
        #222;

}


.container {

    max-width:
        1200px;

    margin:
        0 auto;

    padding:
        30px;

}


h1 {

    margin-bottom:
        5px;

}


.subtitle {

    color:
        #666;

    margin-bottom:
        25px;

}


.camera-controls {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        10px;

    align-items:
        center;

    margin-bottom:
        20px;

}


select {

    padding:
        10px;

    border:
        1px solid #ccc;

    border-radius:
        6px;

    min-width:
        250px;

}


button {

    border:
        none;

    padding:
        11px 18px;

    border-radius:
        6px;

    cursor:
        pointer;

    font-size:
        14px;

}


#refreshCamera {

    background:
        #6c757d;

    color:
        white;

}


#startCamera {

    background:
        #198754;

    color:
        white;

}


#stopCamera {

    background:
        #dc3545;

    color:
        white;

}


#startSession {

    background:
        #0d6efd;

    color:
        white;

}


#endSession {

    background:
        #fd7e14;

    color:
        white;

}


button:disabled {

    opacity:
        0.5;

    cursor:
        not-allowed;

}


.status {

    padding:
        12px;

    background:
        white;

    border-radius:
        8px;

    margin-bottom:
        20px;

    border:
        1px solid #ddd;

}


.session-info {

    display:
        grid;

    grid-template-columns:
        repeat(
            3,
            1fr
        );

    gap:
        15px;

    margin-bottom:
        20px;

}


.card {

    background:
        white;

    padding:
        18px;

    border-radius:
        8px;

    border:
        1px solid #ddd;

}


.card-title {

    font-size:
        13px;

    color:
        #777;

    margin-bottom:
        7px;

}


.card-value {

    font-size:
        20px;

    font-weight:
        bold;

}


.viewer {

    width:
        100%;

    background:
        black;

    border-radius:
        10px;

    overflow:
        hidden;

    position:
        relative;

}


#video {

    display:
        block;

    width:
        100%;

    height:
        auto;

}


#overlay {

    position:
        absolute;

    top:
        0;

    left:
        0;

    width:
        100%;

    height:
        100%;

    pointer-events:
        none;

}


.prediction-info {

    margin-top:
        20px;

    background:
        white;

    padding:
        20px;

    border-radius:
        8px;

    border:
        1px solid #ddd;

}


.face-result {

    padding:
        12px;

    margin-bottom:
        10px;

    border:
        1px solid #ddd;

    border-radius:
        6px;

}


.face-result:last-child {

    margin-bottom:
        0;

}


.no-face {

    color:
        #777;

}


@media (
    max-width: 700px
) {

    .session-info {

        grid-template-columns:
            1fr;

    }


    .container {

        padding:
            15px;

    }

}

</style>

</head>


<body>


<div class="container">


<h1>
Real-Time Facial Expression Recognition
</h1>


<div class="subtitle">
Browser Camera → Flask → CNN → Expression
</div>


<!-- ==================================================
     CAMERA CONTROLS
     ================================================== -->

<div class="camera-controls">


<select id="cameraSelect">

<option value="">
Select Camera
</option>

</select>


<button id="refreshCamera">
Refresh Cameras
</button>


<button id="startCamera">
Start Camera
</button>


<button id="stopCamera"
        disabled>
Stop Camera
</button>


<button id="startSession">
Start Session
</button>


<button id="endSession"
        disabled>
End Session
</button>


</div>


<!-- ==================================================
     STATUS
     ================================================== -->

<div class="status"
     id="status">

Camera stopped.

</div>


<!-- ==================================================
     SESSION INFORMATION
     ================================================== -->

<div class="session-info">


<div class="card">

<div class="card-title">
Session
</div>

<div class="card-value"
     id="sessionStatus">

Not Started

</div>

</div>


<div class="card">

<div class="card-title">
Faces Detected
</div>

<div class="card-value"
     id="faceCount">

0

</div>

</div>


<div class="card">

<div class="card-title">
Current Expression
</div>

<div class="card-value"
     id="expression">

-

</div>

</div>


</div>


<!-- ==================================================
     VIDEO VIEWER
     ================================================== -->

<div class="viewer">

<video id="video"
       autoplay
       playsinline>
</video>


<canvas id="overlay">
</canvas>

</div>


<!-- ==================================================
     RESULTS
     ================================================== -->

<div class="prediction-info">

<h3>
Detection Results
</h3>


<div id="results">

<div class="no-face">
No face detected.
</div>

</div>

</div>


</div>


<script>


// ========================================================
// ELEMENTS
// ========================================================

const video =
    document.getElementById(
        "video"
    );


const overlay =
    document.getElementById(
        "overlay"
    );


const overlayContext =
    overlay.getContext(
        "2d"
    );


const cameraSelect =
    document.getElementById(
        "cameraSelect"
    );


const refreshCameraButton =
    document.getElementById(
        "refreshCamera"
    );


const startCameraButton =
    document.getElementById(
        "startCamera"
    );


const stopCameraButton =
    document.getElementById(
        "stopCamera"
    );


const startSessionButton =
    document.getElementById(
        "startSession"
    );


const endSessionButton =
    document.getElementById(
        "endSession"
    );


const statusElement =
    document.getElementById(
        "status"
    );


const sessionStatusElement =
    document.getElementById(
        "sessionStatus"
    );


const faceCountElement =
    document.getElementById(
        "faceCount"
    );


const expressionElement =
    document.getElementById(
        "expression"
    );


const resultsElement =
    document.getElementById(
        "results"
    );


// ========================================================
// VARIABLES
// ========================================================

let stream =
    null;


let predictionInterval =
    null;


let activeSessionId =
    null;


let cameraRunning =
    false;


// --------------------------------------------------------
// IMPORTANT
// --------------------------------------------------------
// Prevents multiple /predict requests from running
// at the same time.
// --------------------------------------------------------

let predictionInProgress =
    false;


// ========================================================
// SMOOTHING
// ========================================================

const SMOOTHING_FRAMES =
    5;


const MAX_FACE_DISTANCE =
    100;


let faceTracks =
    [];


// ========================================================
// CAMERA LIST
// ========================================================

async function loadCameras() {

    try {

        const devices =
            await navigator
                .mediaDevices
                .enumerateDevices();


        cameraSelect.innerHTML =
            "";


        const defaultOption =
            document.createElement(
                "option"
            );


        defaultOption.value =
            "";


        defaultOption.textContent =
            "Select Camera";


        cameraSelect.appendChild(
            defaultOption
        );


        let cameraNumber =
            1;


        devices.forEach(
            function(device) {

                if (
                    device.kind ===
                    "videoinput"
                ) {

                    const option =
                        document.createElement(
                            "option"
                        );


                    option.value =
                        device.deviceId;


                    option.textContent =
                        device.label
                        ||
                        `Camera ${cameraNumber}`;


                    cameraSelect.appendChild(
                        option
                    );


                    cameraNumber++;

                }

            }
        );

    }
    catch (error) {

        console.error(
            "Camera loading error:",
            error
        );

    }

}


// ========================================================
// REFRESH CAMERAS
// ========================================================

refreshCameraButton
    .addEventListener(
        "click",
        async function() {

            await loadCameras();

            statusElement.textContent =
                "Camera list refreshed.";

        }
    );


// ========================================================
// START SESSION
// ========================================================

startSessionButton
    .addEventListener(
        "click",
        async function() {

            if (
                activeSessionId
            ) {

                return;

            }


            try {

                statusElement.textContent =
                    "Starting session...";


                const response =
                    await fetch(
                        "/start_session",
                        {
                            method:
                                "POST"
                        }
                    );


                const result =
                    await response.json();


                if (
                    result.success
                ) {

                    activeSessionId =
                        result.session_id;


                    sessionStatusElement.textContent =
                        "Active";


                    startSessionButton.disabled =
                        true;


                    endSessionButton.disabled =
                        false;


                    statusElement.textContent =
                        "Session started.";

                }
                else {

                    statusElement.textContent =
                        "Could not start session.";

                }

            }
            catch (error) {

                console.error(
                    error
                );


                statusElement.textContent =
                    "Could not start session.";

            }

        }
    );


// ========================================================
// END SESSION
// ========================================================

endSessionButton
    .addEventListener(
        "click",
        async function() {

            if (
                !activeSessionId
            ) {

                return;

            }


            try {

                statusElement.textContent =
                    "Ending session...";


                const response =
                    await fetch(
                        "/end_session",
                        {

                            method:
                                "POST",

                            headers: {

                                "Content-Type":
                                    "application/json"

                            },

                            body:
                                JSON.stringify({

                                    session_id:
                                        activeSessionId

                                })

                        }
                    );


                const result =
                    await response.json();


                if (
                    result.success
                ) {

                    statusElement.textContent =
                        "Session ended. "
                        +
                        "Predictions: "
                        +
                        result.total_predictions
                        +
                        " | Average confidence: "
                        +
                        result.average_confidence.toFixed(
                            2
                        )
                        +
                        "%";


                    activeSessionId =
                        null;


                    sessionStatusElement.textContent =
                        "Not Started";


                    startSessionButton.disabled =
                        false;


                    endSessionButton.disabled =
                        true;

                }
                else {

                    statusElement.textContent =
                        "Could not end session.";

                }

            }
            catch (error) {

                console.error(
                    error
                );


                statusElement.textContent =
                    "Could not end session.";

            }

        }
    );


// ========================================================
// START CAMERA
// ========================================================

startCameraButton
    .addEventListener(
        "click",
        async function() {

            try {

                let selectedDevice =
                    cameraSelect.value;


                const constraints = {

                    video:
                        selectedDevice
                        ?
                        {

                            deviceId:
                            {

                                exact:
                                    selectedDevice

                            }

                        }
                        :
                        true,

                    audio:
                        false

                };


                statusElement.textContent =
                    "Starting camera...";


                stream =
                    await navigator
                        .mediaDevices
                        .getUserMedia(
                            constraints
                        );


                video.srcObject =
                    stream;


                cameraRunning =
                    true;


                predictionInProgress =
                    false;


                startCameraButton.disabled =
                    true;


                stopCameraButton.disabled =
                    false;


                cameraSelect.disabled =
                    true;


                refreshCameraButton.disabled =
                    true;


                await video.play();


                await new Promise(
                    function(resolve) {

                        if (
                            video.readyState
                            >=
                            2
                        ) {

                            resolve();

                        }
                        else {

                            video.onloadedmetadata =
                                function() {

                                    resolve();

                                };

                        }

                    }
                );


                overlay.width =
                    video.videoWidth;


                overlay.height =
                    video.videoHeight;


                faceTracks =
                    [];


                statusElement.textContent =
                    "Camera running. Detecting faces...";


                // ------------------------------------------------
                // PREDICTION LOOP
                // ------------------------------------------------
                //
                // 700 ms gives approximately 1.4 requests/sec.
                //
                // The important part is that we also prevent
                // overlapping requests.
                // ------------------------------------------------

                predictionInterval =
                    setInterval(
                        sendFrameForPrediction,
                        700
                    );

            }
            catch (error) {

                console.error(
                    "Camera error:",
                    error
                );


                statusElement.textContent =
                    "Could not access camera: "
                    +
                    error.message;

            }

        }
    );


// ========================================================
// STOP CAMERA
// ========================================================

stopCameraButton
    .addEventListener(
        "click",
        function() {

            stopCamera();

        }
    );


// ========================================================
// STOP CAMERA FUNCTION
// ========================================================

function stopCamera() {

    if (
        predictionInterval
    ) {

        clearInterval(
            predictionInterval
        );


        predictionInterval =
            null;

    }


    predictionInProgress =
        false;


    if (
        stream
    ) {

        stream
            .getTracks()
            .forEach(
                function(track) {

                    track.stop();

                }
            );


        stream =
            null;

    }


    video.srcObject =
        null;


    overlayContext.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    cameraRunning =
        false;


    startCameraButton.disabled =
        false;


    stopCameraButton.disabled =
        true;


    cameraSelect.disabled =
        false;


    refreshCameraButton.disabled =
        false;


    faceTracks =
        [];


    faceCountElement.textContent =
        "0";


    expressionElement.textContent =
        "-";


    resultsElement.innerHTML =
        '<div class="no-face">Camera stopped.</div>';


    statusElement.textContent =
        "Camera stopped.";

}


// ========================================================
// CAPTURE FRAME
// ========================================================

function captureFrame() {

    const canvas =
        document.createElement(
            "canvas"
        );


    // ------------------------------------------------------
    // Send a smaller frame to Render.
    //
    // This reduces upload size and server processing.
    // ------------------------------------------------------

    let width =
        video.videoWidth;


    let height =
        video.videoHeight;


    const maxWidth =
        640;


    if (
        width
        >
        maxWidth
    ) {

        const scale =
            maxWidth
            /
            width;


        width =
            maxWidth;


        height =
            Math.round(
                height
                *
                scale
            );

    }


    canvas.width =
        width;


    canvas.height =
        height;


    const context =
        canvas.getContext(
            "2d"
        );


    context.drawImage(
        video,
        0,
        0,
        width,
        height
    );


    return canvas.toDataURL(
        "image/jpeg",
        0.65
    );

}


// ========================================================
// FACE DISTANCE
// ========================================================

function faceDistance(
    face1,
    face2
) {

    const centerX1 =
        face1.x
        +
        face1.width
        /
        2;


    const centerY1 =
        face1.y
        +
        face1.height
        /
        2;


    const centerX2 =
        face2.x
        +
        face2.width
        /
        2;


    const centerY2 =
        face2.y
        +
        face2.height
        /
        2;


    return Math.sqrt(

        Math.pow(
            centerX1
            -
            centerX2,
            2
        )

        +

        Math.pow(
            centerY1
            -
            centerY2,
            2
        )

    );

}


// ========================================================
// UPDATE FACE TRACKS
// ========================================================

function updateFaceTracks(
    detectedFaces
) {

    const newTracks =
        [];


    const usedTracks =
        new Set();


    detectedFaces.forEach(
        function(face) {

            let bestTrack =
                null;


            let bestDistance =
                Infinity;


            faceTracks.forEach(
                function(track) {

                    if (
                        usedTracks.has(
                            track.id
                        )
                    ) {

                        return;

                    }


                    const distance =
                        faceDistance(
                            face,
                            track.face
                        );


                    if (
                        distance
                        <
                        bestDistance
                        &&
                        distance
                        <
                        MAX_FACE_DISTANCE
                    ) {

                        bestDistance =
                            distance;


                        bestTrack =
                            track;

                    }

                }
            );


            if (
                bestTrack
            ) {

                usedTracks.add(
                    bestTrack.id
                );


                bestTrack.face =
                    face;


                bestTrack.history.push({

                    expression:
                        face.expression,

                    confidence:
                        face.confidence

                });


                if (
                    bestTrack.history.length
                    >
                    SMOOTHING_FRAMES
                ) {

                    bestTrack.history.shift();

                }


                bestTrack.missingFrames =
                    0;


                newTracks.push(
                    bestTrack
                );

            }
            else {

                const newTrack = {

                    id:
                        Date.now()
                        +
                        Math.random(),

                    face:
                        face,

                    history: [

                        {

                            expression:
                                face.expression,

                            confidence:
                                face.confidence

                        }

                    ],

                    missingFrames:
                        0

                };


                newTracks.push(
                    newTrack
                );

            }

        }
    );


    faceTracks.forEach(
        function(track) {

            if (
                !usedTracks.has(
                    track.id
                )
            ) {

                track.missingFrames++;

            }

        }
    );


    faceTracks.forEach(
        function(track) {

            if (
                track.missingFrames
                <
                5
            ) {

                if (
                    !newTracks.some(
                        function(newTrack) {

                            return (
                                newTrack.id
                                ===
                                track.id
                            );

                        }
                    )
                ) {

                    newTracks.push(
                        track
                    );

                }

            }

        }
    );


    faceTracks =
        newTracks;


    return faceTracks;

}


// ========================================================
// GET SMOOTHED RESULT
// ========================================================

function getSmoothedResult(
    history
) {

    const counts =
        {};


    history.forEach(
        function(item) {

            if (
                !counts[
                    item.expression
                ]
            ) {

                counts[
                    item.expression
                ] =
                    0;

            }


            counts[
                item.expression
            ]++;

        }
    );


    let bestExpression =
        history[
            history.length - 1
        ].expression;


    let bestCount =
        0;


    Object.keys(
        counts
    ).forEach(
        function(expression) {

            if (
                counts[
                    expression
                ]
                >
                bestCount
            ) {

                bestCount =
                    counts[
                        expression
                    ];


                bestExpression =
                    expression;

            }

        }
    );


    const matchingConfidences =
        history
            .filter(
                function(item) {

                    return (
                        item.expression
                        ===
                        bestExpression
                    );

                }
            )
            .map(
                function(item) {

                    return (
                        item.confidence
                    );

                }
            );


    const averageConfidence =
        matchingConfidences.reduce(
            function(
                sum,
                value
            ) {

                return (
                    sum
                    +
                    value
                );

            },
            0
        )
        /
        matchingConfidences.length;


    return {

        expression:
            bestExpression,

        confidence:
            averageConfidence

    };

}


// ========================================================
// SEND FRAME TO SERVER
// ========================================================

async function sendFrameForPrediction() {

    if (
        !cameraRunning
    ) {

        return;

    }


    // ------------------------------------------------------
    // CRITICAL PERFORMANCE FIX
    //
    // If the previous request is still running,
    // DO NOT send another request.
    //
    // This prevents Render from getting overloaded.
    // ------------------------------------------------------

    if (
        predictionInProgress
    ) {

        return;

    }


    predictionInProgress =
        true;


    try {

        const imageData =
            captureFrame();


        const response =
            await fetch(
                "/predict",
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            image:
                                imageData,

                            session_id:
                                activeSessionId

                        })

                }
            );


        const result =
            await response.json();


        if (
            !result.success
        ) {

            console.error(
                result.message
            );


            statusElement.textContent =
                "Prediction error: "
                +
                result.message;


            return;

        }


        // =================================================
        // FACE COUNT
        // =================================================

        faceCountElement.textContent =
            result.faces;


        // =================================================
        // NO FACE
        // =================================================

        if (
            result.faces
            ===
            0
        ) {

            expressionElement.textContent =
                "-";


            resultsElement.innerHTML =
                '<div class="no-face">No face detected.</div>';


            faceTracks =
                [];


            overlayContext.clearRect(
                0,
                0,
                overlay.width,
                overlay.height
            );


            statusElement.textContent =
                "Camera running. No face detected.";


            return;

        }


        // =================================================
        // FACE TRACKING / SMOOTHING
        // =================================================

        const tracks =
            updateFaceTracks(
                result.faces_data
            );


        // =================================================
        // OVERLAY SIZE
        // =================================================

        overlay.width =
            video.videoWidth;


        overlay.height =
            video.videoHeight;


        overlayContext.clearRect(
            0,
            0,
            overlay.width,
            overlay.height
        );


        // =================================================
        // RESULT HTML
        // =================================================

        let resultHTML =
            "";


        let firstExpression =
            "-";


        tracks.forEach(
            function(track) {

                const smoothed =
                    getSmoothedResult(
                        track.history
                    );


                const face =
                    track.face;


                if (
                    firstExpression
                    ===
                    "-"
                ) {

                    firstExpression =
                        smoothed.expression;

                }


                // -----------------------------------------
                // FACE RECTANGLE
                // -----------------------------------------

                overlayContext.strokeStyle =
                    "#00ff00";


                overlayContext.lineWidth =
                    3;


                overlayContext.strokeRect(

                    face.x,

                    face.y,

                    face.width,

                    face.height

                );


                // -----------------------------------------
                // LABEL
                // -----------------------------------------

                const label =
                    smoothed.expression
                    +
                    " "
                    +
                    smoothed.confidence.toFixed(
                        1
                    )
                    +
                    "%";


                overlayContext.font =
                    "bold 18px Arial";


                const textWidth =
                    overlayContext.measureText(
                        label
                    ).width;


                let labelX =
                    face.x;


                let labelY =
                    face.y
                    -
                    8;


                if (
                    labelY
                    <
                    25
                ) {

                    labelY =
                        face.y
                        +
                        face.height
                        +
                        25;

                }


                overlayContext.fillStyle =
                    "rgba(0, 0, 0, 0.7)";


                overlayContext.fillRect(

                    labelX,

                    labelY - 20,

                    textWidth + 10,

                    25

                );


                overlayContext.fillStyle =
                    "#00ff00";


                overlayContext.fillText(

                    label,

                    labelX + 5,

                    labelY

                );


                // -----------------------------------------
                // RESULT CARD
                // -----------------------------------------

                resultHTML += `

                    <div class="face-result">

                        <strong>
                            Face
                        </strong>

                        <br>

                        Expression:
                        <strong>
                            ${smoothed.expression}
                        </strong>

                        <br>

                        Confidence:
                        <strong>
                            ${smoothed.confidence.toFixed(2)}%
                        </strong>

                    </div>

                `;

            }
        );


        expressionElement.textContent =
            firstExpression;


        resultsElement.innerHTML =
            resultHTML;


        statusElement.textContent =
            "Camera running. Face detection active.";


    }
    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        statusElement.textContent =
            "Prediction request failed.";

    }
    finally {

        // --------------------------------------------------
        // VERY IMPORTANT
        // --------------------------------------------------
        // Allow the next request only after the previous
        // request has completely finished.
        // --------------------------------------------------

        predictionInProgress =
            false;

    }

}


// ========================================================
// INITIAL LOAD
// ========================================================

window.addEventListener(
    "load",
    async function() {

        await loadCameras();

    }
);


// ========================================================
// PAGE CLOSE
// ========================================================

window.addEventListener(
    "beforeunload",
    function() {

        stopCamera();

    }
);


</script>


</body>

</html>
"""


# ==========================================================
# CREATE DATABASE
# ==========================================================

create_database()


# ==========================================================
# RUN FLASK SERVER
# ==========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("=" * 60)
    print("ONLINE FACIAL EXPRESSION RECOGNITION")
    print("=" * 60)

    print()
    print(
        "Open in browser:"
    )

    print(
        f"http://127.0.0.1:{port}"
    )

    print()
    print(
        "Server running..."
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )