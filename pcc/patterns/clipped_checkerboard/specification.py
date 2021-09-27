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
from ..common import paper_format_str
from ..export import svgwrite2image

_logger = logging.getLogger('ClippedCheckerboard')

@dataclass
class ClippedCheckerboardSpecification(object):
    """This class encapsulates the parameters of a clipped checkerboard
calibration board, where the first & last rows/columns contain clipped
cells. Thus, given an NxM specification, the board will have (N-1) full
squares along each row, with a leading and trailing "half square":
  -------------
  |  xx  xx  x
  | x  xx  xx
  | x  xx  xx
  |  xx  xx  x
  |  xx  xx  x
       ... 

*** Adjustable Parameters ***
name:   Identifier of this calibration pattern

num_squares_horizontal, num_squares_vertical: Number of squares along the
        corresponding dimension

checkerboard_square_length_mm: side length of a checkerboard square in [mm]
    
color_background, color_foreground: SVG color used to fill the background, specify via:
        * named colors: white, red, orange, ...
        * hex color string: #ff9e2c
        * rgb color string: rgb(255, 128, 44)

overlay_board_specifications: Flag to enable/disable overlay of the board
        specification. If enabled, the parametrization will be printed within
        the board's bottom margin (if there is enough space).

*** Computed Parameters ***
...
board_width_mm, board_height_mm: Dimensions of the physical board in [mm]
"""
    
    name: str
    num_squares_horizontal: int
    num_squares_vertical: int
    checkerboard_square_length_mm: int
    
    color_background: str = 'white'
    color_foreground: str = 'black'

    overlay_board_specifications: bool = True

    board_width_mm: int = field(init=False)
    board_height_mm: int = field(init=False)
    reference_points: np.ndarray = field(init=False)

    def __post_init__(self):
        """Derives remaining attributes after user intialization."""
        self.board_width_mm = (self.num_squares_horizontal + 1) * self.checkerboard_square_length_mm
        self.board_height_mm = (self.num_squares_vertical + 1) * self.checkerboard_square_length_mm
        #TODO ref points
    
    def __repr__(self) -> str:
        return f'[pcc] ClippedCheckerboard: {paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} a {self.checkerboard_square_length_mm}mm'

    def svg(self) -> svgwrite.Drawing:
        """Returns the SVG drawing of this calibration board."""
        _logger.info(f'Drawing calibration pattern: {self}')
        
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

        cb = dwg.add(dwg.g(id='checkerboard'))
        square_length_half_mm = self.checkerboard_square_length_mm / 2
        for row in range(self.num_squares_vertical + 1):
            if row in [0, self.num_squares_vertical]:
                # Top- and bottom-most rows contain "half squares"
                top = square_length_half_mm if row == 0 else row * self.checkerboard_square_length_mm
                height = square_length_half_mm
            else:
                # All other rows contain "full squares"
                height = self.checkerboard_square_length_mm
                top = row * self.checkerboard_square_length_mm
            for col in range((row + 1) % 2, self.num_squares_horizontal + 1, 2):
                if col in [0, self.num_squares_horizontal]:
                    # Left- and right-most columns contain "half squares"
                    left = square_length_half_mm if col == 0 else col * self.checkerboard_square_length_mm
                    width = square_length_half_mm
                else:
                    # All other columns contain "full squares"
                    left = col * self.checkerboard_square_length_mm
                    width = self.checkerboard_square_length_mm
                cb.add(dwg.rect(insert=(_mm(left), _mm(top)),
                                size=(_mm(width), _mm(height)),
                                class_="pattern"))
        
        # Overlay pattern information
        if self.overlay_board_specifications:
            font_size_mm = 4
            line_padding_mm = 1
            text_height_mm = 2 * (font_size_mm + line_padding_mm)
            overlay_color = 'rgb(120, 120, 120)'
            available_space = square_length_half_mm * 0.6
            # If we don't have enough space, try adding only a single line of text:
            single_line = text_height_mm > available_space
            if single_line:
                text_height_mm = font_size_mm + line_padding_mm
            
            if available_space < text_height_mm:
                _logger.warning(f'Cannot overlay specification. Available free space {available_space}mm is too small (requiring at least {text_height_mm} mm).')
            else:
                top = min(self.board_height_mm - available_space, self.board_height_mm - text_height_mm) + font_size_mm
                overlay = dwg.add(dwg.g(style=f"font-size:{_mm(font_size_mm)};font-family:monospace;stroke:{overlay_color};stroke-width:1;fill:{overlay_color};"))
                if single_line:
                    overlay.add(dwg.text(f'pcc::ClippedCheckerboard {paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} \u00E0 {self.checkerboard_square_length_mm}mm',
                                         insert=(_mm(square_length_half_mm / 2), _mm(top))))
                else:
                    overlay.add(dwg.text('pcc::ClippedCheckerboard', insert=(_mm(square_length_half_mm / 2), _mm(top))))
                    top += font_size_mm + line_padding_mm
                    overlay.add(dwg.text(f'{paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} \u00E0 {self.checkerboard_square_length_mm}mm',
                                        insert=(_mm(square_length_half_mm / 2), _mm(top))))
                # dwg.add(overlay)
        return dwg

    def image(self) -> np.ndarray:
        """Renders the calibration pattern to an image (NumPy ndarray)."""
        return svgwrite2image(self.svg())
