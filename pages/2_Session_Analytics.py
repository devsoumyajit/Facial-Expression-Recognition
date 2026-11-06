import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "expression_recognition.db"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="🙂",
    layout="wide"
)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def load_predictions():

    if not os.path.exists(
        DATABASE_PATH
    ):
        return pd.DataFrame()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    query = """
        SELECT
            id,
            session_id,
            timestamp,
            expression,
            confidence,
            faces_detected
        FROM predictions
        ORDER BY id
    """

    dataframe = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return dataframe


def load_sessions():

    if not os.path.exists(
        DATABASE_PATH
    ):
        return pd.DataFrame()

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    query = """
        SELECT
            id,
            session_id,
            start_time,
            end_time,
            duration_seconds,
            total_predictions,
            most_detected_expression,
            average_confidence,
            average_faces_detected
        FROM sessions
        ORDER BY id DESC
    """

    dataframe = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return dataframe


# ============================================================
# LOAD DATA
# ============================================================

predictions_df = load_predictions()
sessions_df = load_sessions()


# ============================================================
# TITLE
# ============================================================

st.title(
    "Real-Time Facial Expression Recognition"
)

st.subheader(
    "Session Analytics Dashboard"
)

st.write(
    "Deep Learning based facial expression "
    "recognition and analysis system."
)


# ============================================================
# DATABASE CHECK
# ============================================================

if predictions_df.empty:

    st.warning(
        "No prediction data found in the database."
    )

    st.stop()


# ============================================================
# SESSION FILTER
# ============================================================

st.markdown(
    "### Session Selection"
)

session_options = [
    "All Sessions"
]

session_options.extend(
    sessions_df[
        "session_id"
    ].tolist()
)

selected_session = st.selectbox(
    "Select a session to analyse",
    session_options
)


# ============================================================
# FILTER PREDICTION DATA
# ============================================================

if selected_session == (
    "All Sessions"
):

    filtered_predictions = (
        predictions_df.copy()
    )

else:

    filtered_predictions = (
        predictions_df[
            predictions_df[
                "session_id"
            ] == selected_session
        ].copy()
    )


# ============================================================
# NO DATA CHECK
# ============================================================

if filtered_predictions.empty:

    st.warning(
        "No prediction data available "
        "for this session."
    )

    st.stop()


# ============================================================
# BASIC METRICS
# ============================================================

total_predictions = (
    len(
        filtered_predictions
    )
)

total_sessions = (
    filtered_predictions[
        "session_id"
    ].nunique()
)

average_confidence = (
    filtered_predictions[
        "confidence"
    ].mean()
)

average_faces = (
    filtered_predictions[
        "faces_detected"
    ].mean()
)

most_detected_expression = (
    filtered_predictions[
        "expression"
    ]
    .value_counts()
    .idxmax()
)


# ============================================================
# OVERALL STATISTICS
# ============================================================

st.markdown(
    "### Overall Statistics"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Predictions",
        total_predictions
    )

with col2:

    st.metric(
        "Sessions",
        total_sessions
    )

with col3:

    st.metric(
        "Average Confidence",
        f"{average_confidence:.2f}%"
    )

with col4:

    st.metric(
        "Most Detected",
        most_detected_expression
    )


# ============================================================
# SESSION DETAILS
# ============================================================

if selected_session != (
    "All Sessions"
):

    selected_session_data = (
        sessions_df[
            sessions_df[
                "session_id"
            ] == selected_session
        ]
    )

    if not selected_session_data.empty:

        session = (
            selected_session_data.iloc[0]
        )

        st.markdown(
            "### Selected Session Details"
        )

        detail_col1, detail_col2, \
        detail_col3, detail_col4 = (
            st.columns(4)
        )

        with detail_col1:

            st.metric(
                "Start Time",
                session[
                    "start_time"
                ]
            )

        with detail_col2:

            st.metric(
                "End Time",
                session[
                    "end_time"
                ]
            )

        with detail_col3:

            st.metric(
                "Duration",
                f"{session['duration_seconds']:.2f} sec"
            )

        with detail_col4:

            st.metric(
                "Average Faces",
                f"{session['average_faces_detected']:.2f}"
            )


# ============================================================
# EXPRESSION DISTRIBUTION
# ============================================================

st.markdown(
    "### Expression Distribution"
)

expression_counts = (
    filtered_predictions[
        "expression"
    ]
    .value_counts()
    .reset_index()
)

expression_counts.columns = [
    "Expression",
    "Count"
]

fig_expression = px.bar(
    expression_counts,
    x="Expression",
    y="Count",
    text="Count",
    title="Detected Facial Expressions"
)

fig_expression.update_layout(
    xaxis_title="Expression",
    yaxis_title="Number of Predictions"
)

st.plotly_chart(
    fig_expression,
    use_container_width=True
)


# ============================================================
# EXPRESSION PERCENTAGE
# ============================================================

st.markdown(
    "### Expression Percentage"
)

percentage_data = (
    filtered_predictions[
        "expression"
    ]
    .value_counts()
    .reset_index()
)

percentage_data.columns = [
    "Expression",
    "Count"
]

fig_pie = px.pie(
    percentage_data,
    names="Expression",
    values="Count",
    title="Expression Distribution"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

st.markdown(
    "### Confidence Analysis"
)

fig_confidence = px.line(
    filtered_predictions,
    x="id",
    y="confidence",
    markers=True,
    title="Prediction Confidence"
)

fig_confidence.update_layout(
    xaxis_title="Prediction ID",
    yaxis_title="Confidence (%)"
)

st.plotly_chart(
    fig_confidence,
    use_container_width=True
)


# ============================================================
# FACE DETECTION ANALYSIS
# ============================================================

st.markdown(
    "### Face Detection Analysis"
)

fig_faces = px.line(
    filtered_predictions,
    x="id",
    y="faces_detected",
    markers=True,
    title="Faces Detected"
)

fig_faces.update_layout(
    xaxis_title="Prediction ID",
    yaxis_title="Number of Faces"
)

st.plotly_chart(
    fig_faces,
    use_container_width=True
)


# ============================================================
# SESSION TABLE
# ============================================================

st.markdown(
    "### Session Information"
)

if not sessions_df.empty:

    display_sessions = (
        sessions_df.copy()
    )

    display_sessions[
        "duration_seconds"
    ] = (
        display_sessions[
            "duration_seconds"
        ].round(2)
    )

    display_sessions[
        "average_confidence"
    ] = (
        display_sessions[
            "average_confidence"
        ].round(2)
    )

    display_sessions[
        "average_faces_detected"
    ] = (
        display_sessions[
            "average_faces_detected"
        ].round(2)
    )

    display_sessions = (
        display_sessions.rename(
            columns={
                "session_id":
                    "Session ID",

                "start_time":
                    "Start Time",

                "end_time":
                    "End Time",

                "duration_seconds":
                    "Duration (seconds)",

                "total_predictions":
                    "Predictions",

                "most_detected_expression":
                    "Most Detected",

                "average_confidence":
                    "Avg Confidence (%)",

                "average_faces_detected":
                    "Avg Faces"
            }
        )
    )

    display_columns = [
        "Session ID",
        "Start Time",
        "End Time",
        "Duration (seconds)",
        "Predictions",
        "Most Detected",
        "Avg Confidence (%)",
        "Avg Faces"
    ]

    st.dataframe(
        display_sessions[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "View Raw Prediction Data"
):

    st.dataframe(
        filtered_predictions,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Real-Time Facial Expression Recognition "
    "and Analysis System Using Deep Learning"
)