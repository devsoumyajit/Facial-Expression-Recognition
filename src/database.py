import os
import sqlite3
from datetime import datetime


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "expression_recognition.db"
)


def create_database():

    os.makedirs(
        DATABASE_DIR,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    # ------------------------------------------------
    # PREDICTIONS TABLE
    # ------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            expression TEXT NOT NULL,
            confidence REAL NOT NULL,
            faces_detected INTEGER NOT NULL
        )
    """)

    # ------------------------------------------------
    # SESSION SUMMARY TABLE
    # ------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_seconds REAL,
            total_predictions INTEGER DEFAULT 0,
            most_detected_expression TEXT,
            average_confidence REAL DEFAULT 0,
            average_faces_detected REAL DEFAULT 0
        )
    """)

    connection.commit()
    connection.close()

    print("Database created successfully.")
    print(
        f"Database path: {DATABASE_PATH}"
    )


def insert_prediction(
    session_id,
    expression,
    confidence,
    faces_detected
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO predictions
        (
            session_id,
            timestamp,
            expression,
            confidence,
            faces_detected
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        str(session_id),
        str(timestamp),
        str(expression),
        float(confidence),
        int(faces_detected)
    ))

    connection.commit()
    connection.close()


def start_session(session_id):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    start_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT OR IGNORE INTO sessions
        (
            session_id,
            start_time
        )
        VALUES (?, ?)
    """, (
        str(session_id),
        start_time
    ))

    connection.commit()
    connection.close()


def end_session(session_id):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    # ------------------------------------------------
    # GET SESSION PREDICTIONS
    # ------------------------------------------------

    cursor.execute("""
        SELECT
            timestamp,
            expression,
            confidence,
            faces_detected
        FROM predictions
        WHERE session_id = ?
        ORDER BY id
    """, (
        str(session_id),
    ))

    predictions = cursor.fetchall()

    if len(predictions) == 0:

        connection.close()

        return

    # ------------------------------------------------
    # SESSION TIMES
    # ------------------------------------------------

    start_time = (
        predictions[0][0]
    )

    end_time = (
        predictions[-1][0]
    )

    start_datetime = datetime.strptime(
        start_time,
        "%Y-%m-%d %H:%M:%S"
    )

    end_datetime = datetime.strptime(
        end_time,
        "%Y-%m-%d %H:%M:%S"
    )

    duration_seconds = (
        end_datetime
        - start_datetime
    ).total_seconds()

    # ------------------------------------------------
    # CALCULATE ANALYTICS
    # ------------------------------------------------

    expressions = [
        prediction[1]
        for prediction in predictions
    ]

    confidences = [
        float(prediction[2])
        for prediction in predictions
    ]

    faces = [
        int(prediction[3])
        for prediction in predictions
    ]

    expression_counts = {}

    for expression in expressions:

        if expression not in expression_counts:

            expression_counts[
                expression
            ] = 0

        expression_counts[
            expression
        ] += 1

    most_detected_expression = max(
        expression_counts,
        key=expression_counts.get
    )

    average_confidence = (
        sum(confidences)
        / len(confidences)
    )

    average_faces_detected = (
        sum(faces)
        / len(faces)
    )

    total_predictions = (
        len(predictions)
    )

    # ------------------------------------------------
    # UPDATE SESSION
    # ------------------------------------------------

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
        end_time,
        float(duration_seconds),
        int(total_predictions),
        str(most_detected_expression),
        float(average_confidence),
        float(average_faces_detected),
        str(session_id)
    ))

    connection.commit()
    connection.close()


def get_predictions():

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


def get_sessions():

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


def main():

    print("DATABASE TEST")
    print("=" * 50)

    create_database()

    test_session_id = (
        "TEST_SESSION_001"
    )

    print("\nSTARTING TEST SESSION")

    start_session(
        test_session_id
    )

    print(
        "Test session started."
    )

    print(
        "\nINSERTING TEST PREDICTION"
    )

    insert_prediction(
        session_id=test_session_id,
        expression="Happy",
        confidence=92.50,
        faces_detected=1
    )

    print(
        "Test prediction inserted successfully."
    )

    print(
        "\nENDING TEST SESSION"
    )

    end_session(
        test_session_id
    )

    print(
        "Test session ended."
    )

    print(
        "\nSAVED PREDICTIONS"
    )

    print("=" * 50)

    predictions = get_predictions()

    for prediction in predictions:

        print(
            f"ID: {prediction[0]}"
        )

        print(
            f"Session ID: {prediction[1]}"
        )

        print(
            f"Timestamp: {prediction[2]}"
        )

        print(
            f"Expression: {prediction[3]}"
        )

        print(
            f"Confidence: "
            f"{float(prediction[4]):.2f}%"
        )

        print(
            f"Faces detected: "
            f"{int(prediction[5])}"
        )

        print("-" * 50)

    print(
        "\nSESSION SUMMARIES"
    )

    print("=" * 50)

    sessions = get_sessions()

    for session in sessions:

        print(
            f"Session ID: {session[1]}"
        )

        print(
            f"Start time: {session[2]}"
        )

        print(
            f"End time: {session[3]}"
        )

        print(
            f"Duration: "
            f"{float(session[4]):.2f} seconds"
        )

        print(
            f"Total predictions: "
            f"{int(session[5])}"
        )

        print(
            f"Most detected expression: "
            f"{session[6]}"
        )

        print(
            f"Average confidence: "
            f"{float(session[7]):.2f}%"
        )

        print(
            f"Average faces detected: "
            f"{float(session[8]):.2f}"
        )

        print("-" * 50)


if __name__ == "__main__":

    main()