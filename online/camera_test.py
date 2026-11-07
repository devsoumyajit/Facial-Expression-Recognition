import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Camera Test",
    page_icon="📷",
    layout="wide"
)

st.title("Browser Camera Test")

st.write(
    "This page tests whether the browser can access "
    "the camera connected to the current computer."
)

components.html(
    """
    <!DOCTYPE html>
    <html>

    <head>
        <style>

            body {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                text-align: center;
            }

            video {
                width: 100%;
                max-width: 900px;
                border-radius: 12px;
                border: 2px solid #444;
                background: black;
            }

            button {
                margin-top: 15px;
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }

            #status {
                margin-top: 15px;
                font-size: 16px;
            }

        </style>
    </head>

    <body>

        <video
            id="camera"
            autoplay
            playsinline
            muted>
        </video>

        <br>

        <button onclick="startCamera()">
            Start Camera
        </button>

        <p id="status">
            Camera not started.
        </p>

        <script>

            async function startCamera() {

                const video =
                    document.getElementById("camera");

                const status =
                    document.getElementById("status");

                try {

                    const stream =
                        await navigator.mediaDevices
                        .getUserMedia({
                            video: true,
                            audio: false
                        });

                    video.srcObject = stream;

                    status.innerHTML =
                        "Camera started successfully.";

                }

                catch (error) {

                    status.innerHTML =
                        "Camera access failed: "
                        + error.message;

                    console.error(error);

                }
            }

        </script>

    </body>

    </html>
    """,
    height=700
)