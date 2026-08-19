import sqlite3
import config
from datetime import datetime

def get_connection():
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                gender TEXT,
                income_type TEXT,
                annual_income REAL,
                employment_years REAL,
                education_level TEXT,
                employment_status TEXT,
                existing_loan_balance REAL,
                credit_inquiries INTEGER,
                credit_history_years REAL,
                payment_status TEXT,
                prediction INTEGER,
                prediction_label TEXT,
                approval_probability REAL,
                model_used TEXT
            )
        ''')
        conn.commit()

def save_prediction(input_data: dict, result: dict):
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, gender, income_type, annual_income, employment_years,
                education_level, employment_status, existing_loan_balance,
                credit_inquiries, credit_history_years, payment_status,
                prediction, prediction_label, approval_probability, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_data.get("gender", "N/A"),
            input_data.get("income_type", "N/A"),
            float(input_data.get("annual_income", 0)),
            float(input_data.get("employment_years", 0)),
            input_data.get("education_level", "N/A"),
            input_data.get("employment_status", "N/A"),
            float(input_data.get("existing_loan_balance", 0)),
            int(input_data.get("credit_inquiries", 0)),
            float(input_data.get("credit_history_years", 0)),
            input_data.get("payment_status", "N/A"),
            int(result.get("prediction", 0)),
            result.get("label", "Rejected"),
            float(result.get("approval_probability", 0.0)),
            result.get("model", "Machine Learning Model")
        ))
        conn.commit()

def get_history(limit=100, search_query=None):
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        if search_query:
            q = f"%{search_query}%"
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE gender LIKE ? OR income_type LIKE ? OR education_level LIKE ? OR employment_status LIKE ? OR prediction_label LIKE ?
                ORDER BY id DESC LIMIT ?
            ''', (q, q, q, q, q, limit))
        else:
            cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def get_dashboard_stats():
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as total FROM predictions')
        total = cursor.fetchone()['total']

        cursor.execute('SELECT COUNT(*) as approved FROM predictions WHERE prediction = 1')
        approved = cursor.fetchone()['approved']

        cursor.execute('SELECT COUNT(*) as rejected FROM predictions WHERE prediction = 0')
        rejected = cursor.fetchone()['rejected']

        approval_rate = round((approved / total * 100), 2) if total > 0 else 0.0

        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approval_rate
        }
