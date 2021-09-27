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
from ..export import svgwrite2image

@dataclass
class DAIBoardSpecification(object):
    """This class encapsulates the parameters of a checkerboard calibration
board. As all 'board specification' classes, it provides functionality to
render this board as SVG or PNG. To locate this board in images, refer to
the submodule's detection submodule.


*** Adjustable Parameters ***
name:   Identifier of this calibration pattern

board_width_mm, board_height_mm: Dimensions of the physical board in [mm]

margin_horizontal_mm, margin_vertical_mm: TODO (what about the clipped 1st row/col?)

checkerboard_square_length_mm: side length of a (full) checkerboard square in [mm]
    
color_background, color_foreground: SVG color used to fill the background, specify via:
    * named colors: white, red, orange, ...
    * hex color string: #ff9e2c
    * rgb color string: rgb(255, 128, 44)


*** Computed Parameters ***
num_squares_horizontal, num_squares_vertical: TODO
    """
    
    name: str
    board_width_mm: int
    board_height_mm: int
    margin_horizontal_mm: int
    margin_vertical_mm: int
    checkerboard_square_length_mm: int
    
    color_background: str = 'white'
    color_foreground: str = 'black'

    num_squares_horizontal: int = field(init=False)
    num_squares_vertical: int = field(init=False)

    def __post_init__(self):
        """Derives remaining attributes after user intialization."""
        self.num_squares_horizontal = (self.board_width_mm - 2*self.margin_horizontal_mm) // self.checkerboard_square_length_mm
        self.num_squares_vertical = (self.board_height_mm - 2*self.margin_vertical_mm) // self.checkerboard_square_length_mm
        # Notify user of configuration issues
        tmp = self.num_squares_horizontal * self.checkerboard_square_length_mm + 2*self.margin_horizontal_mm
        if tmp != self.board_width_mm:
            logging.error(f'MISMATCH: Board width {self.board_width_mm} vs pattern {tmp}!')
        tmp = self.num_squares_vertical * self.checkerboard_square_length_mm + 2*self.margin_vertical_mm
        if tmp != self.board_height_mm:
            logging.error(f'MISMATCH: Board height {self.board_height_mm} vs pattern {tmp}!')

        # print(self.num_squares_horizontal, 'x', self.num_squares_vertical)


    def svg(self) -> svgwrite.Drawing:
        """Returns the SVG drawing of this calibration board."""
        logging.info(f'Drawing calibration pattern: {self}')
        
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

        #TODO remove dummy overlay (to indicate margins)
        dwg.add(dwg.rect(insert=(_mm(self.margin_horizontal_mm), _mm(self.margin_vertical_mm)),
                         size=(_mm(self.board_width_mm - 2*self.margin_horizontal_mm), _mm(self.board_height_mm - 2*self.margin_vertical_mm)),
                         fill='rgb(180,180,180)'))

        # Draw checkerboard (TODO first & last were clipped, iirc)
        cb = dwg.add(dwg.g(id='checkerboard'))
        top = self.margin_vertical_mm
        for row in range(self.num_squares_vertical + 1):
            if row in [0, self.num_squares_vertical]:
                height = self.checkerboard_square_length_mm / 2
                # top = self.margin_vertical_mm
            else:
                # top = self.margin_vertical_mm + row * self.checkerboard_square_length_mm
                height = self.checkerboard_square_length_mm

            left = self.margin_horizontal_mm
            for col in range(self.num_squares_horizontal + 1):
                if col in [0, self.num_squares_horizontal]:
                    width = self.checkerboard_square_length_mm / 2
                else:
                    width = self.checkerboard_square_length_mm
                if col % 2 == row % 2: # We only have to insert the foreground colored squares
                    cb.add(dwg.rect(insert=(_mm(left), _mm(top)),
                                    size=(_mm(width), _mm(height)),
                                    class_="pattern"))
                left += width
            # columns = range(0, self.num_squares_horizontal, 2) if row % 2 == 0 else range(1, self.num_squares_horizontal, 2)
            # for col in columns:
            #     left = self.margin_horizontal_mm + col * self.checkerboard_square_length_mm
            #     cb.add(dwg.rect(insert=(_mm(left), _mm(top)),
            #                     size=(_mm(self.checkerboard_square_length_mm), _mm(height)),
            #                     class_="pattern"))
            top += height
        return dwg

    def image(self) -> np.ndarray:
        """Renders the calibration pattern to an image (NumPy ndarray)."""
        return svgwrite2image(self.svg())
