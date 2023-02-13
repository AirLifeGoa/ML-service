"""Class to log exceptions for all Model Factory related issues."""
# import logging

# log = logging.getLogger(__name__)


class ModelTypeError(Exception):
    """Check Error in model type for model factory."""

    def __init__(self, message="Model type not found"):
        """
        Initialize ModelTypeError.

        :param message: message for the error log
        """
        super().__init__(message)

class ModelDataTypeError(Exception):
    """Check Error in model type for model factory."""

    def __init__(self, message="Model type not found"):
        """
        Initialize ModelTypeError.

        :param message: message for the error log
        """
        super().__init__(message)


# class ModelNotFoundException(Exception):
#     """Model not found in filesystem."""

#     def __init__(self):
#         """Initialize ModelNotFoundException."""
#         log.error("Model not found in filesystem")


# class ScalerNotFoundException(Exception):
#     """Scaler not found in filesystem."""

#     def __init__(self):
#         """Initialize ScalerException."""
#         log.error("Scaler not found in filesystem")
