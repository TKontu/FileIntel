"""
DEPRECATED: Validation functions migrated to fileintel.core.validation.

This module is maintained for backward compatibility but all new validation
should use the centralized validation module.
"""

from typing import List
from fastapi import UploadFile
from .error_handlers import APIErrorHandler
from ..core.validation import validate_file_upload, FileValidationError


def validate_content_type(file: UploadFile, supported_formats: List[str]):
    """
    DEPRECATED: Use fileintel.core.validation.validate_file_upload instead.
    Validate the content type of the uploaded file.
    """
    try:
        validate_file_upload(file, supported_formats)
    except FileValidationError as e:
        raise APIErrorHandler.unsupported_file_type(
            file.content_type, supported_formats
        )
