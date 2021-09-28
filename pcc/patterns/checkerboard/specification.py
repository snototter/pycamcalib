import logging
from pcc.patterns.common import SpecificationError
import svgwrite
import numpy as np
from dataclasses import dataclass, field
from ..svgutils import svgwrite2image, overlay_pattern_specification
from ..common import paper_format_str


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

checkerboard_square_length_mm: side length of a checkerboard square in [mm]

num_squares_horizontal, num_squares_vertical: Number of squares along the
        corresponding dimension

color_background, color_foreground, color_overlay: SVG color, specify as either
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
    color_overlay: str = 'rgb(120, 120, 120)'

    overlay_board_specifications: bool = True

    margin_horizontal_mm: float = field(init=False)
    margin_vertical_mm: float = field(init=False)
    reference_points: np.ndarray = field(init=False, repr=False)

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
        self.reference_points = np.zeros((self.num_inner_corners_horizontal * self.num_inner_corners_vertical, 3), np.float32)
        v, u = np.meshgrid(np.arange(self.num_inner_corners_vertical), np.arange(self.num_inner_corners_horizontal))
        self.reference_points[:, 0] = u.flatten() * self.checkerboard_square_length_mm
        self.reference_points[:, 1] = v.flatten() * self.checkerboard_square_length_mm

    @property
    def num_inner_corners_horizontal(self):
        return self.num_squares_horizontal - 1

    @property
    def num_inner_corners_vertical(self):
        return self.num_squares_vertical - 1

    def svg(self) -> svgwrite.Drawing:
        """Returns the SVG drawing of this calibration board."""
        _logger.info(f'Drawing calibration pattern: {self.name}')

        # Helper to put fully-specified coordinates (in millimeters)
        def _mm(v):
            return f'{v}mm'

        dwg = svgwrite.Drawing(profile='full', height=_mm(self.board_height_mm), width=_mm(self.board_width_mm), debug=False)
        # Height/width weren't set properly in the c'tor (all export tests had
        # 100% height/width instead of the desired metric dimensions. Thus, we
        # have to reset the attributes after construction:
        dwg.attribs['height'] = _mm(self.board_height_mm)
        dwg.attribs['width'] = _mm(self.board_width_mm)

        # Define CSS styles
        dwg.defs.add(dwg.style(f".pattern {{ fill: {self.color_foreground}; stroke: none; }}"))

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
            fmt_str = f'{paper_format_str(self.board_width_mm, self.board_height_mm)}, {self.num_squares_horizontal}x{self.num_squares_vertical} \u00E0 {self.checkerboard_square_length_mm}mm, margins: {self.margin_horizontal_mm:.1f}mm x {self.margin_vertical_mm:.1f}mm'
            overlay_pattern_specification(dwg, 'pcc::Checkerboard', fmt_str,
                                          board_height_mm=self.board_height_mm,
                                          available_space_mm=self.margin_vertical_mm * 0.6,
                                          offset_left_mm=self.margin_horizontal_mm / 2,
                                          color_overlay=self.color_overlay)
        return dwg

    def image(self) -> np.ndarray:
        """Renders the calibration pattern to an image (NumPy ndarray)."""
        return svgwrite2image(self.svg())
