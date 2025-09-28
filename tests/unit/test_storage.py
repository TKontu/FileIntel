import unittest
from unittest.mock import MagicMock, patch
from src.fileintel.storage.postgresql_storage import PostgreSQLStorage
from src.fileintel.storage.models import Document, Job, Result
import uuid


class TestPostgreSQLStorage(unittest.TestCase):
    def setUp(self):
        self.mock_db_session = MagicMock()
        self.storage = PostgreSQLStorage(self.mock_db_session)

    def test_create_document(self):
        doc_id = str(uuid.uuid4())
        with patch("uuid.uuid4", return_value=doc_id):
            document = self.storage.create_document(
                filename="test.pdf",
                content_hash="12345",
                file_size=1024,
                mime_type="application/pdf",
            )
        self.mock_db_session.add.assert_called_once()
        self.mock_db_session.commit.assert_called_once()
        self.mock_db_session.refresh.assert_called_once()
        self.assertEqual(document.id, doc_id)
        self.assertEqual(document.filename, "test.pdf")

    def test_save_job(self):
        job_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "document_id": doc_id,
            "status": "pending",
            "data": {"key": "value"},
        }
        self.storage.save_job(job_data)
        self.mock_db_session.add.assert_called_once()
        self.mock_db_session.commit.assert_called_once()

    def test_get_job(self):
        job_id = str(uuid.uuid4())
        mock_job = Job(id=job_id, status="pending")
        self.mock_db_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )
        job = self.storage.get_job(job_id)
        self.assertEqual(job, mock_job)

    def test_update_job_status(self):
        job_id = str(uuid.uuid4())
        mock_job = Job(id=job_id, status="pending")
        self.mock_db_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )
        self.storage.update_job_status(job_id, "completed")
        self.assertEqual(mock_job.status, "completed")
        self.mock_db_session.commit.assert_called_once()

    def test_save_result(self):
        job_id = str(uuid.uuid4())
        result_data = {"result": "success"}
        self.storage.save_result(job_id, result_data)
        self.mock_db_session.add.assert_called_once()
        self.mock_db_session.commit.assert_called_once()

    def test_get_result(self):
        job_id = str(uuid.uuid4())
        mock_result = Result(job_id=job_id, data='{"result": "success"}')
        self.mock_db_session.query.return_value.filter.return_value.first.return_value = (
            mock_result
        )
        result = self.storage.get_result(job_id)
        self.assertEqual(result, mock_result)


if __name__ == "__main__":
    unittest.main()
