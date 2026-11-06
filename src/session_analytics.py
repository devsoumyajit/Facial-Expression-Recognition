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


def calculate_analytics(predictions):

    if len(predictions) == 0:

        print("No prediction data available.")

        return

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

    sessions = {
        prediction[1]
        for prediction in predictions
    }

    expression_counts = Counter(
        expressions
    )

    total_predictions = (
        len(predictions)
    )

    total_sessions = (
        len(sessions)
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
        expression_counts.most_common(1)[0][0]
    )

    print()
    print("=" * 60)
    print("SESSION ANALYTICS")
    print("=" * 60)

    print(
        f"Total Predictions       : "
        f"{total_predictions}"
    )

    print(
        f"Total Sessions          : "
        f"{total_sessions}"
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

    print()
    print("=" * 60)
    print("SESSION LIST")
    print("=" * 60)

    for session_id in sorted(
        sessions
    ):

        session_records = [
            prediction
            for prediction in predictions
            if prediction[1] == session_id
        ]

        session_expressions = [
            prediction[3]
            for prediction in session_records
        ]

        session_confidences = [
            float(prediction[4])
            for prediction in session_records
        ]

        session_most_common = (
            Counter(
                session_expressions
            ).most_common(1)[0][0]
        )

        session_average_confidence = (
            sum(session_confidences)
            / len(session_confidences)
        )

        print()
        print(
            f"Session ID          : "
            f"{session_id}"
        )

        print(
            f"Predictions         : "
            f"{len(session_records)}"
        )

        print(
            f"Most Detected       : "
            f"{session_most_common}"
        )

        print(
            f"Average Confidence  : "
            f"{session_average_confidence:.2f}%"
        )


def main():

    print("FACIAL EXPRESSION ANALYTICS")

    predictions = load_predictions()

    calculate_analytics(
        predictions
    )


if __name__ == "__main__":

    main()