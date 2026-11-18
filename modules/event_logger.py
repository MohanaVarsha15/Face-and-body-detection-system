
import sqlite3
from pathlib import Path
from datetime import datetime


class EventLogger:

    def __init__(self):

        # Project root / data / monitoring_history.db
        project_root = Path(__file__).resolve().parent.parent

        self.data_directory = project_root / "data"
        self.data_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.db_path = (
            self.data_directory / "monitoring_history.db"
        )

        self.connection = sqlite3.connect(
            str(self.db_path),
            timeout=10
        )

        self.connection.execute("PRAGMA journal_mode=WAL")

        self.last_status = {
            "person": None,
            "eyes": None
        }

        self.create_table()

        print(
            f"Monitoring database ready: {self.db_path}"
        )

    def create_table(self):

        query = """
        CREATE TABLE IF NOT EXISTS monitoring_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL,
            description TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """

        self.connection.execute(query)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_timestamp
            ON monitoring_events(timestamp)
        """)

        self.connection.commit()

    def record_status_change(self, event_type, new_status):

        allowed_statuses = {
            "person": [
                "PERSON DETECTED",
                "NO PERSON DETECTED"
            ],
            "eyes": [
                "OPEN",
                "CLOSED"
            ]
        }

        # Ignore unknown or unsupported statuses.
        if event_type not in allowed_statuses:
            return False

        if new_status not in allowed_statuses[event_type]:
            return False

        previous_status = self.last_status[event_type]

        # Do not store repeated statuses.
        if previous_status == new_status:
            return False

        # Update the previous status.
        self.last_status[event_type] = new_status

        # Generate a readable event description.
        if event_type == "person":

            if new_status == "PERSON DETECTED":
                description = "A person entered the camera view."
            else:
                description = "No person detected in the camera view."

        else:

            if new_status == "OPEN":
                description = "Eyes changed to OPEN."
            else:
                description = "Eyes changed to CLOSED."

        # Local timestamp with timezone information.
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        query = """
        INSERT INTO monitoring_events
        (event_type, status, description, timestamp)
        VALUES (?, ?, ?, ?)
        """

        try:

            self.connection.execute(
                query,
                (
                    event_type,
                    new_status,
                    description,
                    timestamp
                )
            )

            self.connection.commit()

            print(
                f"[EVENT SAVED] {event_type.upper()}: "
                f"{new_status} | {timestamp}"
            )

            return True

        except sqlite3.Error as error:

            print("Database logging error:", error)

            return False

    def close(self):

        if self.connection:
            self.connection.close()
            print("Monitoring database connection closed.")
