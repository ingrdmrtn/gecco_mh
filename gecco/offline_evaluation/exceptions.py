from pydantic import ValidationError
from typing import Optional, Dict, Any


class ModelValidationError(Exception):
    """Base exception for model validation failures with structured details."""

    def __init__(
        self,
        message: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}


class CodeSafetyError(ModelValidationError):
    """Raised when code contains forbidden patterns or invalid syntax."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CodeSafetyError", details)


class ParameterMismatchError(ModelValidationError):
    """Raised when declared parameters don't match code extraction."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ParameterMismatchError", details)


class PydanticSchemaError(ModelValidationError):
    """Raised when Pydantic schema validation fails."""

    def __init__(self, pydantic_error: ValidationError):
        errors = pydantic_error.errors()
        messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err["loc"])
            messages.append(f"{loc}: {err['msg']}")

        super().__init__(
            message="Schema validation failed",
            error_type="PydanticSchemaError",
            details={"validation_errors": messages, "raw_errors": errors},
        )
