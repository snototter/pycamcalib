import logging
import io
from pcc.patterns.common import SpecificationError
import svgwrite
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from dataclasses import dataclass, field
from vito import imutils
from ..export import svgwrite2image
from ..common import paper_format_str
# from collections import deque
# from ..common import GridIndex, Rect, Point, sort_points_ccw, center, SpecificationError

_logger = logging.getLogger('Checkerboard')

@dataclass
class CheckerboardSpecification(object):
    """This class encapsulates the parameters of a standard checkerboard
calibration board. As all 'board specification' classes, it provides functionality
to render this board as SVG or PNG. To locate this board in images, refer to
the submodule's detection submodule.


*** Adjustable Parameters ***
name:   Identifier of this calibration pattern

board_width_mm, board_height_mm: Dimensions of the physical board in [mm]

checkerboard_square_length_mm: side length of a (full) checkerboard square in [mm]

num_squares_horizontal, num_squares_vertical: Number of squares along the
        corresponding dimension
    
color_background, color_foreground: SVG color used to fill the background, specify via:
        * named colors: white, red, orange, ...
        * hex color string: #ff9e2c
        * rgb color string: rgb(255, 128, 44)

overlay_board_specifications: Flag to enable/disable overlay of the board
        specification. If enabled, the parametrization will be printed within
        the board's bottom margin (if there is enough space).


*** Computed Parameters ***
margin_horizontal_mm, margin_vertical_mm: Margins between checkerboard pattern
        and the board's edge.

reference_points: Object points in 3d to be used as reference/correspondences
        for calibration.
    """
    
    name: str
    board_width_mm: int
    board_height_mm: int
    checkerboard_square_length_mm: int
    num_squares_horizontal: int
    num_squares_vertical: int
    
    color_background: str = 'white'
    color_foreground: str = 'black'

    overlay_board_specifications: bool = True

    margin_horizontal_mm: int = field(init=False)
    margin_vertical_mm: int = field(init=False)
    reference_points: np.ndarray = field(init=False)

    def __post_init__(self):
        """Derives uninitialized attributes and performs sanity checks."""
        self.margin_horizontal_mm = (self.board_width_mm - self.num_squares_horizontal * self.checkerboard_square_length_mm) / 2
        self.margin_vertical_mm = (self.board_height_mm - self.num_squares_vertical * self.checkerboard_square_length_mm) / 2
        # Sanity checks
        if self.margin_horizontal_mm < 0:
            raise SpecificationError('Horizontal margin < 0 (too many squares per row).')
        if self.margin_vertical_mm < 0:
            raise SpecificationError('Vertical margin < 0 (too many squares per column).')
        # Set 3d object points (only consider INNER corners)
        inner_rows = self.num_squares_vertical - 1
        inner_cols = self.num_squares_horizontal - 1
        self.reference_points = np.zeros((inner_cols * inner_rows, 3), np.float32)
        self.reference_points[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2) * self.checkerboard_square_length_mm

    def __repr__(self) -> str:
        return f'[pcc] Checkerboard: {paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} a {self.checkerboard_square_length_mm}mm'

    def svg(self) -> svgwrite.Drawing:
        """Returns the SVG drawing of this calibration board."""
        _logger.info(f'Drawing calibration pattern: {self.name}')
        
        # Helper to put fully-specified coordinates (in millimeters)
        def _mm(v):
            return f'{v}mm'

        dwg = svgwrite.Drawing(profile='full')
        #, height=f'{h_target_mm}mm', width=f'{w_target_mm}mm', profile='tiny', debug=False)
        # Height/width weren't set properly in the c'tor (my SVGs had 100% instead
        # of the desired dimensions). Thus, we set the attributes manually:
        dwg.attribs['height'] = _mm(self.board_height_mm)
        dwg.attribs['width'] = _mm(self.board_width_mm)

        dwg.defs.add(dwg.style(f".pattern {{ stroke: {self.color_foreground}; stroke-width:1px; }}"))

        # Background should not be transparent
        dwg.add(dwg.rect(insert=(0, 0), size=(_mm(self.board_width_mm), _mm(self.board_height_mm)), fill=self.color_background))

        # Draw checkerboard
        cb = dwg.add(dwg.g(id='checkerboard'))
        for row in range(self.num_squares_vertical):
            top = self.margin_vertical_mm + row * self.checkerboard_square_length_mm

            for col in range(row % 2, self.num_squares_horizontal, 2):  # Ensures alternating placement
                left = self.margin_horizontal_mm + col * self.checkerboard_square_length_mm
                cb.add(dwg.rect(insert=(_mm(left), _mm(top)),
                                size=(_mm(self.checkerboard_square_length_mm), _mm(self.checkerboard_square_length_mm)),
                                class_="pattern"))
        
        # Overlay pattern information
        if self.overlay_board_specifications:
            font_size_mm = 4
            line_padding_mm = 1
            text_height_mm = 2 * (font_size_mm + line_padding_mm)
            overlay_color = 'rgb(120, 120, 120)'
            if self.margin_vertical_mm < text_height_mm:
                _logger.warning(f'Cannot overlay specification. Bottom margin {self.margin_vertical_mm}mm is too small (requiring a min. of {text_height_mm} mm).')
            else:
                top = min(self.board_height_mm - self.margin_vertical_mm / 4, self.board_height_mm - text_height_mm)
                overlay = dwg.g(style=f"font-size:{_mm(font_size_mm)};font-family:monospace;stroke:{overlay_color};stroke-width:1;fill:{overlay_color};")
                overlay.add(dwg.text('pcc::Checkerboard', insert=(_mm(self.margin_horizontal_mm / 4), _mm(top))))
                top += font_size_mm + line_padding_mm
                overlay.add(dwg.text(f'{paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} \u00E0 {self.checkerboard_square_length_mm}mm',
                                     insert=(_mm(self.margin_horizontal_mm / 4), _mm(top))))
                dwg.add(overlay)

        return dwg

    def image(self) -> np.ndarray:
        """Renders the calibration pattern to an image (NumPy ndarray)."""
        return svgwrite2image(self.svg())
