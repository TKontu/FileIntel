class DocumentAnalyzerException(Exception):
    """Base exception for the document analyzer application."""
    pass

class ConfigException(DocumentAnalyzerException):
    """Exception related to configuration errors."""
    pass

class DocumentProcessingException(DocumentAnalyzerException):
    """Exception related to document processing errors."""
    pass

class LLMException(DocumentAnalyzerException):
    """Exception related to LLM provider errors."""
    pass

class APIException(DocumentAnalyzerException):
    """Exception related to API errors."""
    pass

class StorageException(DocumentAnalyzerException):
    """Exception related to storage errors."""
    pass
