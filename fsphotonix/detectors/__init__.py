"""Detection screens (plugin kind ``"detector"``)."""

from .base import Detector, DetectorOutput, as_detector
from .camera import CameraArray
from .fiber import CustomDetector, MMFArray, SMFArray
from .photodiode import PhotodiodeArray, PowerMeter

__all__ = [
    "Detector", "DetectorOutput", "as_detector",
    "CameraArray", "PhotodiodeArray", "PowerMeter",
    "SMFArray", "MMFArray", "CustomDetector",
]
