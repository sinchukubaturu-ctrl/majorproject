import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Database Connection
# ----------------------------------------------------------
def get_connection():
    return sqlite3.connect("arrg_db.sqlite", check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS patient_records (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            age TEXT,
            gender TEXT,
            date_of_scan TEXT,
            symptoms TEXT,
            family_history TEXT,
            head_injury_notes TEXT,
            other_conditions TEXT,
            image_filename TEXT,
            pred_label TEXT,
            confidence REAL,
            dicom_filename TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()

# ----------------------------------------------------------
# Fetch All Patients
# ----------------------------------------------------------
def get_all_patients():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patient_records ORDER BY timestamp ASC;")
    records = cur.fetchall()
    cur.close()
    conn.close()
    return records
