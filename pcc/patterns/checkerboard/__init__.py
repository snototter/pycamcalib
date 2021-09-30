"""
Checkerboard calibration pattern.
See, e.g. https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html
"""
from .specification_standard import CheckerboardSpecification
from .specification_shifted import ShiftedCheckerboardSpecification
from .detection import CheckerboardDetector
