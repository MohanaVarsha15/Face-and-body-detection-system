
import sqlite3
from pathlib import Path

db_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "monitoring_history.db"
)

connection = sqlite3.connect(str(db_path))

cursor = connection.cursor()

cursor.execute("""
    SELECT id, event_type, status, description, timestamp
    FROM monitoring_events
    ORDER BY id DESC
""")

records = cursor.fetchall()

print("\n========== MONITORING HISTORY ==========\n")

if not records:
    print("No monitoring events recorded.")

else:

    for record in records:

        print("-" * 60)
        print("Event ID   :", record[0])
        print("Event Type :", record[1])
        print("Status     :", record[2])
        print("Description:", record[3])
        print("Timestamp  :", record[4])

    print("-" * 60)

connection.close()
