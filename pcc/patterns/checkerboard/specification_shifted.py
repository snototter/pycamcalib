import logging
import svgwrite
import numpy as np
from dataclasses import dataclass, field
from ..common import paper_format_str, SpecificationError
from ..svgutils import svgwrite2image, overlay_pattern_specification


_logger = logging.getLogger('ShiftedCheckerboard')


@dataclass
class ShiftedCheckerboardSpecification(object):
    """This class encapsulates the parameters of a shifted checkerboard
calibration board, where the first & last rows/columns contain "half
squares". Thus, if you specify N squares per row, the board will have (N-1)
full squares along each row, with a leading and trailing "half square".
Consequently, this board will have N x M inner corners if N = number
of squares per row and M = number of squares per column.

*** Adjustable Parameters ***
name:   Identifier of this calibration pattern

board_width_mm, board_height_mm: Dimensions of the physical board in [mm]

num_squares_horizontal, num_squares_vertical: Number of squares along the
        corresponding dimension

checkerboard_square_length_mm: side length of a checkerboard square in [mm]

color_background, color_foreground, color_overlay: SVG color, specify as either
        * named colors: white, red, orange, ...
        * hex color string: #ff9e2c
        * rgb color string: rgb(255, 128, 44)

overlay_board_specifications: Flag to enable/disable overlay of the board
        specification. If enabled, the parametrization will be printed within
        the board's bottom margin (if there is enough space).

*** Computed Parameters ***

margin_horizontal_mm, margin_vertical_mm: Distance from the edge of the physical
        board to the closest outer square edge.
"""

    name: str
    board_width_mm: int
    board_height_mm: int
    num_squares_horizontal: int
    num_squares_vertical: int
    checkerboard_square_length_mm: int

    color_background: str = 'white'
    color_foreground: str = 'black'
    color_overlay: str = 'rgb(120, 120, 120)'

    overlay_board_specifications: bool = True

    margin_horizontal_mm: int = field(init=False)
    margin_vertical_mm: int = field(init=False)
    reference_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Derives remaining attributes after user intialization."""
        self.margin_horizontal_mm = (self.board_width_mm - self.num_squares_horizontal * self.checkerboard_square_length_mm) / 2
        self.margin_vertical_mm = (self.board_height_mm - self.num_squares_vertical * self.checkerboard_square_length_mm) / 2
        # Sanity checks
        if self.margin_horizontal_mm < 0:
            raise SpecificationError(f'Horizontal margin {self.margin_horizontal_mm} < 0 (too many squares per row). Check specification for {self}')
        if self.margin_vertical_mm < 0:
            raise SpecificationError(f'Vertical margin {self.margin_vertical_mm} < 0 (too many squares per column). Check specification for {self}')
        # Set 3d object points (only consider INNER corners)
        self.reference_points = np.zeros((self.num_inner_corners_horizontal * self.num_inner_corners_vertical, 3), np.float32)
        v, u = np.meshgrid(np.arange(self.num_inner_corners_vertical), np.arange(self.num_inner_corners_horizontal))
        self.reference_points[:, 0] = u.flatten() * self.checkerboard_square_length_mm
        self.reference_points[:, 1] = v.flatten() * self.checkerboard_square_length_mm

    @property
    def num_inner_corners_horizontal(self):
        return self.num_squares_horizontal

    @property
    def num_inner_corners_vertical(self):
        return self.num_squares_vertical

    def svg(self) -> svgwrite.Drawing:
        """Returns the SVG drawing of this calibration board."""
        _logger.info(f'Drawing calibration pattern: {self.name}')

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
                top = self.margin_vertical_mm if row == 0 else\
                      (self.board_height_mm - self.margin_vertical_mm - square_length_half_mm)
                height = square_length_half_mm
            else:
                # All other rows contain "full squares"
                height = self.checkerboard_square_length_mm
                top = self.margin_vertical_mm + square_length_half_mm\
                    + (row - 1) * self.checkerboard_square_length_mm
            for col in range((row + 1) % 2, self.num_squares_horizontal + 1, 2):
                if col in [0, self.num_squares_horizontal]:
                    # Left- and right-most columns contain "half squares"
                    left = self.margin_horizontal_mm if col == 0 else\
                           (self.board_width_mm - self.margin_horizontal_mm - square_length_half_mm)
                    width = square_length_half_mm
                else:
                    # All other columns contain "full squares"
                    left = self.margin_horizontal_mm + square_length_half_mm\
                         + (col - 1) * self.checkerboard_square_length_mm
                    width = self.checkerboard_square_length_mm
                cb.add(dwg.rect(insert=(_mm(left), _mm(top)),
                                size=(_mm(width), _mm(height)),
                                class_="pattern"))

        # Overlay pattern information
        if self.overlay_board_specifications:
            fmt_str = f'{paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} \u00E0 {self.checkerboard_square_length_mm}mm, margins: {self.margin_horizontal_mm:.1f}mm x {self.margin_vertical_mm:.1f}mm'
            overlay_pattern_specification(dwg, 'pcc::ShiftedCheckerboard', fmt_str,
                                          board_height_mm=self.board_height_mm,
                                          available_space_mm=self.margin_vertical_mm * 0.6,
                                          offset_left_mm=square_length_half_mm/2,
                                          color_overlay=self.color_overlay)
        return dwg

    def image(self) -> np.ndarray:
        """Renders the calibration pattern to an image (NumPy ndarray)."""
        return svgwrite2image(self.svg())
