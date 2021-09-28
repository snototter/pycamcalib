import logging
import io
import svgwrite
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from dataclasses import dataclass, field
from vito import imutils
# from collections import deque
# from ..common import GridIndex, Rect, Point, sort_points_ccw, center, SpecificationError
from ..common import paper_format_str, SpecificationError
from ..svgutils import svgwrite2image, overlay_pattern_specification

_logger = logging.getLogger('ClippedCheckerboard')

@dataclass
class ClippedCheckerboardSpecification(object):
    """This class encapsulates the parameters of a clipped checkerboard
calibration board, where the first & last rows/columns contain clipped
"squares". Thus, if you specify N squares per row, the board will have (N-1)
full squares along each row, with a leading and trailing "half square". For
example, with num_squares_horizontal = 5 the board would look like:
  _____________
  |
  |  xx  xx  x
  | x  xx  xx
  | x  xx  xx
  |  xx  xx  x
  |  xx  xx  x
       ... 
Consequently, this board will have N x M inner corners if N = number
of squares per row and M = number of squares per column.

*** Adjustable Parameters ***
name:   Identifier of this calibration pattern

board_width_mm, board_height_mm: Dimensions of the physical board in [mm]

num_squares_horizontal, num_squares_vertical: Number of squares along the
        corresponding dimension

checkerboard_square_length_mm: side length of a checkerboard square in [mm]
    
color_background, color_foreground: SVG color, specify via:
        * named colors: white, red, orange, ...
        * hex color string: #ff9e2c
        * rgb color string: rgb(255, 128, 44)

overlay_board_specifications: Flag to enable/disable overlay of the board
        specification. If enabled, the parametrization will be printed within
        the board's bottom margin (if there is enough space).

*** Computed Parameters ***

margin_horizontal_mm, margin_vertical_mm: Distance from the edge of the physical
        board to the closest outer square edge.

TODO double-check doc before release
"""
    
    name: str
    board_width_mm: int
    board_height_mm: int
    num_squares_horizontal: int
    num_squares_vertical: int
    checkerboard_square_length_mm: int
    
    color_background: str = 'white'
    color_foreground: str = 'black'

    overlay_board_specifications: bool = True

    margin_horizontal_mm: int = field(init=False)
    margin_vertical_mm: int = field(init=False)
    reference_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Derives remaining attributes after user intialization."""
        # self.board_width_mm = (self.num_squares_horizontal + 1) * self.checkerboard_square_length_mm
        # self.board_height_mm = (self.num_squares_vertical + 1) * self.checkerboard_square_length_mm

        self.margin_horizontal_mm = (self.board_width_mm - self.num_squares_horizontal * self.checkerboard_square_length_mm) / 2
        self.margin_vertical_mm = (self.board_height_mm - self.num_squares_vertical * self.checkerboard_square_length_mm) / 2
        # Sanity checks
        if self.margin_horizontal_mm < 0:
            raise SpecificationError(f'Horizontal margin {self.margin_horizontal_mm} < 0 (too many squares per row). Check specification for {self}')
        if self.margin_vertical_mm < 0:
            raise SpecificationError(f'Vertical margin {self.margin_vertical_mm} < 0 (too many squares per column). Check specification for {self}')
        # Set 3d object points (only consider INNER corners)
        #FIXME check
        inner_rows = self.num_squares_vertical - 1
        inner_cols = self.num_squares_horizontal - 1
        self.reference_points = np.zeros((inner_cols * inner_rows, 3), np.float32)
        self.reference_points[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2) * self.checkerboard_square_length_mm
    
    # def __repr__(self) -> str:
    #     return f'[pcc] ClippedCheckerboard: {paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} a {self.checkerboard_square_length_mm}mm'

    def svg(self) -> svgwrite.Drawing:
        """Returns the SVG drawing of this calibration board."""
        _logger.info(f'Drawing calibration pattern: {self.name}')
        print('TODO', self.__repr__())
        
        # Helper to put fully-specified coordinates (in millimeters)
        def _mm(v):
            return f'{v}mm'

        dwg = svgwrite.Drawing(profile='full')
        # Height/width must be set after c'tor to ensure correct dimensions/units
        dwg.attribs['height'] = _mm(self.board_height_mm)
        dwg.attribs['width'] = _mm(self.board_width_mm)

        # Define CSS styles
        dwg.defs.add(dwg.style(f".pattern {{ fill: {self.color_foreground}; stroke: none; }}"))

        # Background should not be transparent
        dwg.add(dwg.rect(insert=(0, 0), size=(_mm(self.board_width_mm), _mm(self.board_height_mm)), fill=self.color_background))

        cb = dwg.add(dwg.g(id='checkerboard'))
        square_length_half_mm = self.checkerboard_square_length_mm / 2
        for row in range(self.num_squares_vertical + 1):
            if row in [0, self.num_squares_vertical]:
                # Top- and bottom-most rows contain "half squares"
                top = self.margin_vertical_mm if row == 0 else (self.board_height_mm - self.margin_vertical_mm - square_length_half_mm)
                height = square_length_half_mm
            else:
                # All other rows contain "full squares"
                height = self.checkerboard_square_length_mm
                top = self.margin_vertical_mm + square_length_half_mm + (row - 1) * self.checkerboard_square_length_mm
            for col in range((row + 1) % 2, self.num_squares_horizontal + 1, 2):
                if col in [0, self.num_squares_horizontal]:
                    # Left- and right-most columns contain "half squares"
                    left = self.margin_horizontal_mm if col == 0 else (self.board_width_mm - self.margin_horizontal_mm - square_length_half_mm)
                    width = square_length_half_mm
                else:
                    # All other columns contain "full squares"
                    left = self.margin_horizontal_mm + square_length_half_mm + (col - 1) * self.checkerboard_square_length_mm
                    width = self.checkerboard_square_length_mm
                cb.add(dwg.rect(insert=(_mm(left), _mm(top)),
                                size=(_mm(width), _mm(height)),
                                class_="pattern"))
        
        # Overlay pattern information
        if self.overlay_board_specifications:
            fmt_str = f'{paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} \u00E0 {self.checkerboard_square_length_mm}mm, margins: {self.margin_horizontal_mm:.1f}mm x {self.margin_vertical_mm:.1f}mm'
            overlay_pattern_specification(dwg, 'pcc::ClippedCheckerboard', fmt_str,
                                          board_height_mm=self.board_height_mm,
                                          available_space_mm=self.margin_vertical_mm * 0.6,
                                          offset_left_mm=square_length_half_mm/2)
        return dwg

    def image(self) -> np.ndarray:
        """Renders the calibration pattern to an image (NumPy ndarray)."""
        return svgwrite2image(self.svg())
