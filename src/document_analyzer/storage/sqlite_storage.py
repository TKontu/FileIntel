import sqlite3
from .base import StorageInterface
from ..core.config import settings

class SQLiteStorage(StorageInterface):
    def __init__(self):
        self.connection = sqlite3.connect(settings.storage.connection_string.replace("sqlite:///", ""))
        self.create_tables()

    def create_tables(self):
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT,
                    data TEXT
                )
            """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    job_id TEXT PRIMARY KEY,
                    data TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                )
            """)

    def save_job(self, job_data: dict) -> str:
        job_id = job_data["id"]
        with self.connection:
            self.connection.execute(
                "INSERT INTO jobs (id, status, data) VALUES (?, ?, ?)",
                (job_id, "pending", str(job_data))
            )
        return job_id

    def get_job(self, job_id: str) -> dict:
        with self.connection:
            cursor = self.connection.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return eval(row[0]) if row else {}

    def update_job_status(self, job_id: str, status: str):
        with self.connection:
            self.connection.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))

    def save_result(self, job_id: str, result_data: dict):
        with self.connection:
            self.connection.execute(
                "INSERT INTO results (job_id, data) VALUES (?, ?)",
                (job_id, str(result_data))
            )

    def get_result(self, job_id: str) -> dict:
        with self.connection:
            cursor = self.connection.execute("SELECT data FROM results WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return eval(row[0]) if row else {}
