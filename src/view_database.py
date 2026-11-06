import os
import sqlite3


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "expression_recognition.db"
)


def view_database():

    print("DATABASE VIEWER")
    print("=" * 60)

    if not os.path.exists(DATABASE_PATH):

        print("Database not found.")

        return

    print(
        f"Database: {DATABASE_PATH}"
    )

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

    print()

    if len(predictions) == 0:

        print("No predictions found.")

        return

    print(
        f"Total records: {len(predictions)}"
    )

    print("=" * 60)

    for prediction in predictions:

        record_id = prediction[0]
        session_id = prediction[1]
        timestamp = prediction[2]
        expression = prediction[3]
        confidence = float(prediction[4])
        faces_detected = int(prediction[5])

        print(
            f"ID            : {record_id}"
        )

        print(
            f"Session ID    : {session_id}"
        )

        print(
            f"Timestamp     : {timestamp}"
        )

        print(
            f"Expression    : {expression}"
        )

        print(
            f"Confidence    : {confidence:.2f}%"
        )

        print(
            f"Faces Detected: {faces_detected}"
        )

        print("-" * 60)


if __name__ == "__main__":

    view_database()