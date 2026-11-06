import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="🙂",
    layout="wide"
)


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "Real-Time Facial Expression Recognition"
)

st.subheader(
    "Deep Learning Based Analysis System"
)

st.write(
    "A real-time facial expression recognition system "
    "using a CNN model, OpenCV and SQLite."
)

st.markdown("---")


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown(
    "## Project Overview"
)

st.write(
    """
    This system detects human faces and recognizes
    seven facial expressions using a trained
    Convolutional Neural Network.
    """
)


# ============================================================
# SUPPORTED EXPRESSIONS
# ============================================================

st.markdown(
    "## Supported Expressions"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("Angry")

with col2:
    st.write("Disgust")

with col3:
    st.write("Fear")

with col4:
    st.write("Happy")


col5, col6, col7 = st.columns(3)

with col5:
    st.write("Neutral")

with col6:
    st.write("Sad")

with col7:
    st.write("Surprise")


# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.markdown(
    "## System Pipeline"
)

st.code(
    """
Webcam / Image
      ↓
Face Detection
      ↓
Grayscale Conversion
      ↓
Resize to 48 × 48
      ↓
Normalization
      ↓
CNN Model
      ↓
Expression + Confidence
      ↓
SQLite Database
      ↓
Session Analytics
    """
)


# ============================================================
# AVAILABLE MODULES
# ============================================================

st.markdown(
    "## Available Modules"
)

st.info(
    "Use the sidebar to open Live Recognition "
    "or Session Analytics."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Real-Time Facial Expression Recognition "
    "and Analysis System Using Deep Learning"
)