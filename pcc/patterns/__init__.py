"""This module encapsulates the supported calibration patterns.
Provides functionality to export patterns for print, detect patterns and
extract point correspondences for calibration.

TODO doc:
Each calibration pattern submodule consists of:
* Specification - parameters defining the calibration board, pattern, colors, etc.
  Provides functionality to render the pattern to SVG, PDF and PNG.
* Detector -
"""
# Import common utils and export functionality for convenience
from pcc.patterns.checkerboard.detection import CheckerboardDetector
from .common import SpecificationError
from .svgutils import export_board

# Import specific patterns only as a submodule
from . import checkerboard
from . import eddie

PATTERNS = {
  'Checkerboard': (checkerboard.CheckerboardSpecification, checkerboard.CheckerboardDetector),
  'Eddie': (None, None) #TODO
}
