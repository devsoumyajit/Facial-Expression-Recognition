import os
import sqlite3
from collections import Counter


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "expression_recognition.db"
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


def load_predictions():

    if not os.path.exists(DATABASE_PATH):

        print("Database not found.")

        return []

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            session_id,
            timestamp,
            expression,
            confidence,
            faces_detected
        FROM predictions
        ORDER BY id
    """)

    predictions = cursor.fetchall()

    connection.close()

    return predictions


def load_sessions():

    if not os.path.exists(DATABASE_PATH):

        print("Database not found.")

        return []

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
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
        ORDER BY id
    """)

    sessions = cursor.fetchall()

    connection.close()

    return sessions


def calculate_analytics(
    predictions,
    sessions
):

    print()
    print("=" * 60)
    print("SESSION ANALYTICS")
    print("=" * 60)


    # ========================================================
    # PREDICTION ANALYTICS
    # ========================================================

    if len(predictions) == 0:

        print(
            "No prediction data available."
        )

    else:

        expressions = [
            prediction[3]
            for prediction in predictions
        ]

        confidences = [
            float(prediction[4])
            for prediction in predictions
        ]

        faces_detected = [
            int(prediction[5])
            for prediction in predictions
        ]

        expression_counts = Counter(
            expressions
        )

        total_predictions = (
            len(predictions)
        )

        # Count unique sessions from predictions
        prediction_sessions = {
            prediction[1]
            for prediction in predictions
        }

        total_prediction_sessions = (
            len(prediction_sessions)
        )

        average_confidence = (
            sum(confidences)
            / total_predictions
        )

        average_faces = (
            sum(faces_detected)
            / total_predictions
        )

        most_detected_expression = (
            expression_counts
            .most_common(1)[0][0]
        )

        print(
            f"Total Predictions       : "
            f"{total_predictions}"
        )

        print(
            f"Total Sessions          : "
            f"{len(sessions)}"
        )

        print(
            f"Sessions With Predictions: "
            f"{total_prediction_sessions}"
        )

        print(
            f"Average Confidence      : "
            f"{average_confidence:.2f}%"
        )

        print(
            f"Average Faces Detected  : "
            f"{average_faces:.2f}"
        )

        print(
            f"Most Detected Expression: "
            f"{most_detected_expression}"
        )


    # ========================================================
    # EXPRESSION ANALYSIS
    # ========================================================

    if len(predictions) > 0:

        print()
        print("=" * 60)
        print("EXPRESSION ANALYSIS")
        print("=" * 60)

        print(
            f"{'Expression':<12}"
            f"{'Count':<10}"
            f"{'Percentage':<12}"
        )

        print("-" * 60)

        expressions = [
            prediction[3]
            for prediction in predictions
        ]

        expression_counts = Counter(
            expressions
        )

        total_predictions = (
            len(predictions)
        )

        for expression in CLASS_NAMES:

            count = expression_counts.get(
                expression,
                0
            )

            percentage = (
                count
                / total_predictions
                * 100
            )

            print(
                f"{expression:<12}"
                f"{count:<10}"
                f"{percentage:.2f}%"
            )

        print("-" * 60)


    # ========================================================
    # SESSION LIST
    # ========================================================

    print()
    print("=" * 60)
    print("SESSION LIST")
    print("=" * 60)


    if len(sessions) == 0:

        print(
            "No sessions available."
        )

        return


    for session in sessions:

        (
            session_db_id,
            session_id,
            start_time,
            end_time,
            duration_seconds,
            total_predictions,
            most_detected_expression,
            average_confidence,
            average_faces_detected
        ) = session


        print()

        print(
            f"Session ID          : "
            f"{session_id}"
        )

        print(
            f"Start Time          : "
            f"{start_time}"
        )

        if end_time:

            print(
                f"End Time            : "
                f"{end_time}"
            )

        else:

            print(
                f"End Time            : "
                f"Active / Not ended"
            )


        if duration_seconds is not None:

            print(
                f"Duration            : "
                f"{float(duration_seconds):.2f} seconds"
            )

        else:

            print(
                f"Duration            : "
                f"N/A"
            )


        print(
            f"Predictions         : "
            f"{total_predictions or 0}"
        )


        if most_detected_expression:

            print(
                f"Most Detected       : "
                f"{most_detected_expression}"
            )

        else:

            print(
                f"Most Detected       : "
                f"N/A"
            )


        if average_confidence is not None:

            print(
                f"Average Confidence  : "
                f"{float(average_confidence):.2f}%"
            )

        else:

            print(
                f"Average Confidence  : "
                f"N/A"
            )


        if average_faces_detected is not None:

            print(
                f"Average Faces       : "
                f"{float(average_faces_detected):.2f}"
            )

        else:

            print(
                f"Average Faces       : "
                f"N/A"
            )

        print("-" * 60)


def main():

    print(
        "FACIAL EXPRESSION ANALYTICS"
    )

    predictions = load_predictions()

    sessions = load_sessions()

    calculate_analytics(
        predictions,
        sessions
    )


if __name__ == "__main__":

    main()