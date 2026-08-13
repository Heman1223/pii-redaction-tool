"""PII Redaction Tool.

Public API:
    from src import redact_document
    redact_document("input.docx", "output/redacted.docx")
"""

from .redactor import RedactionResult, redact_document

__version__ = "0.1.0"
__all__ = ["redact_document", "RedactionResult"]
