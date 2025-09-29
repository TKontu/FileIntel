"""
Base storage infrastructure shared across storage implementations.

Provides common database connection, session management, validation,
and sanitization utilities for all storage classes.
"""

import logging
import re
from typing import Any, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class BaseStorageInfrastructure:
    """
    Shared infrastructure for all storage classes.

    Handles database connections, session management, validation,
    and text sanitization utilities.
    """

    def __init__(self, config_or_session):
        """Initialize with either a config object or database session."""
        # Check if database is disabled (for Flower monitoring)
        import os
        if os.environ.get('DISABLE_DATABASE', 'false').lower() == 'true':
            logger.info("Database disabled via DISABLE_DATABASE environment variable")
            self.db = None
            self.engine = None
            self._owns_session = False
            return

        if hasattr(config_or_session, "query"):
            # It's a database session
            self.db = config_or_session
            self.engine = config_or_session.bind
            self._owns_session = False
        else:
            # It's a config object
            self.config = config_or_session
            database_url = self.config.storage.connection_string
            self.engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_size=10,  # Base pool for concurrent Celery tasks
                max_overflow=20,  # Allow burst connections during heavy processing
                pool_recycle=3600,  # Recycle connections every hour
                pool_timeout=30,  # 30 second timeout for getting connection
            )
            SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            self.db = SessionLocal()
            self._owns_session = True

        self._test_connection()

    def _test_connection(self):
        """Test database connection and log status."""
        # Skip connection test if database is disabled
        if self.db is None:
            logger.info("Database connection test skipped - database disabled")
            return

        try:
            self.db.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _handle_session_error(self, e: Exception):
        """Handle session errors with rollback."""
        logger.error(f"Database session error: {e}")
        try:
            self.db.rollback()
        except Exception as rollback_error:
            logger.error(f"Error during rollback: {rollback_error}")
        raise e

    def _safe_commit(self):
        """Safely commit transaction with error handling."""
        try:
            self.db.commit()
        except SQLAlchemyError as e:
            self._handle_session_error(e)

    def _validate_collection_name(self, name: str) -> str:
        """Validate and sanitize collection name."""
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty")

        name = name.strip()

        # Remove potentially dangerous characters
        if re.search(r'[<>"\'/\\]', name):
            raise ValueError("Collection name contains invalid characters")

        return name

    def _validate_input_security(self, value: str, field_name: str) -> str:
        """Validate input for basic security issues."""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")

        # Check for potential injection patterns
        dangerous_patterns = ["<script", "javascript:", "data:"]
        value_lower = value.lower()

        for pattern in dangerous_patterns:
            if pattern in value_lower:
                raise ValueError(f"{field_name} contains potentially dangerous content")

        return value

    def _clean_text(
        self, text: str, preserve_structure: bool = False, max_length: int = None
    ) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Handle unicode issues
        text = self._handle_unicode_issues(text)

        if not preserve_structure:
            # Basic cleaning for search/analysis
            text = re.sub(r"\s+", " ", text)  # Normalize whitespace
            text = text.strip()

        if max_length and len(text) > max_length:
            text = text[:max_length]

        return text

    def _sanitize_for_llm_prompt(self, text: str) -> str:
        """Sanitize text for safe use in LLM prompts."""
        if not text:
            return ""

        # Remove potentially problematic sequences for LLM prompts
        text = re.sub(r"```.*?```", "[CODE_BLOCK]", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]+`", "[CODE]", text)
        text = re.sub(r"\[INST\].*?\[/INST\]", "[INSTRUCTION]", text, flags=re.DOTALL)

        # Limit length for prompt context
        return self._clean_text(text, max_length=8000)

    def _handle_unicode_issues(self, text: str) -> str:
        """Handle unicode encoding issues in text."""
        if not isinstance(text, str):
            return str(text)

        try:
            # Handle common encoding issues
            text = text.encode("utf-8", errors="ignore").decode("utf-8")

            # Remove or replace problematic unicode characters
            text = re.sub(
                r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F-\u009F]", "", text
            )

            return text
        except Exception as e:
            logger.warning(f"Unicode handling error: {e}")
            return str(text)

    def _clean_result_data(self, data: dict) -> dict:
        """Clean data dictionary for safe output."""
        if not isinstance(data, dict):
            return data

        cleaned = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned[key] = self._clean_text(value)
            elif isinstance(value, dict):
                cleaned[key] = self._clean_result_data(value)
            else:
                cleaned[key] = value

        return cleaned

    def close(self):
        """Close database session if owned by this instance."""
        if self._owns_session and self.db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
