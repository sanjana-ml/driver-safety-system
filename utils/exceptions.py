"""Custom exception hierarchy so every failure mode in the system can be
caught, logged and handled gracefully instead of crashing the app."""


class DriverSafetyError(Exception):
    """Base class for all project-specific exceptions."""


class CameraError(DriverSafetyError):
    """Raised when the webcam cannot be opened or a read fails repeatedly."""


class DatasetError(DriverSafetyError):
    """Raised for missing, empty, or malformed dataset directories."""


class CorruptedImageError(DriverSafetyError):
    """Raised when an image file exists but cannot be decoded."""


class ModelNotFoundError(DriverSafetyError):
    """Raised when a trained model (.h5) file is missing."""


class WeightsNotFoundError(DriverSafetyError):
    """Raised when required model weights (CNN/CBAM weights, etc.) are missing."""


class FaceNotDetectedError(DriverSafetyError):
    """Raised when no face could be located in a frame."""


class MultipleFacesWarning(DriverSafetyError):
    """Raised (and typically caught + logged, not fatal) when more than one
    face is detected; the system falls back to the largest bounding box."""


class InvalidPathError(DriverSafetyError):
    """Raised when a configured path is invalid or inaccessible."""


class PermissionDeniedError(DriverSafetyError):
    """Raised when the process lacks permission to read/write a required path."""


class MissingDependencyError(DriverSafetyError):
    """Raised when a required third-party library is not installed."""


class AudioPlaybackError(DriverSafetyError):
    """Raised when the alarm sound cannot be played."""
