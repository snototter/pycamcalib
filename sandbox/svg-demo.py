import svgwrite
import svglib

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

from dataclasses import dataclass, field

from collections import namedtuple
from functools import lru_cache
GridIndex = namedtuple('GridIndex', 'row col')
Rect = namedtuple('Rect', 'left top width height')

#TODO doc
@dataclass(frozen=True)
class PatternSpecificationEddie:
    """Specification of an Eddie-like calibration pattern as
    we typically use at work.

    Configurable attributes:
        name:               Identifier of this calibration pattern

        target_width_mm:    Width of the calibration target (not the
                            printable/grid area) in [mm]

        target_height_mm:   Height of the calibration target in [mm]

        dia_circles_mm:     Circle diameter in [mm]

        dist_circles_mm:    Center distance between neighboring circles in [mm]

        circles_per_square_edge_length:     Number of circles along each
                                            edge of the square

        circles_per_square_edge_thickness:  Thickness of the square border,
                                            given as a number of circles

        min_margin_target_mm:   Minimum distance between edge of the target
                                and circles, will be adjusted automatically
                                to avoid cropping circles (refer to the
                                "computed_margins" property)

        bg_color:   SVG color used to fill the background, specify via:
                    * named colors: white, red, orange, ...
                    * hex colors: #ff9e2c
                    * rgb colors: rgb(255, 128, 44)

        fg_color:   SVG color used to draw the foreground (circles and
                    center marker), see bg_color

    Calculated attributes:
        r_circles_mm:           Radius of a circle

        circles_per_row:        Number of circles along each row

        circles_per_col:        Number of circles along each column
        skipped_circle_indices: TODO
        square_topleft_corner:  TODO

        marker_size_mm:         Length of the center marker's edge in [mm].

        marker_thickness_mm     Thickness of the center marker's border in [mm].
    """
    # Attributes to be set by the user
    name: str
    target_width_mm: int
    target_height_mm: int
    dia_circles_mm: int
    dist_circles_mm: int
    circles_per_square_edge_length: int = 4
    circles_per_square_edge_thickness: int = 1
    min_margin_target_mm: int = 5
    bg_color: str = 'white'
    fg_color: str = 'black'

    # Automatically calculated attributes
    r_circles_mm : float = field(init=False, repr=False)
    circles_per_row : int = field(init=False, repr=False)
    circles_per_col : int = field(init=False, repr=False)
    skipped_circle_indices: list = field(init=False, repr=False)
    square_topleft_corner: GridIndex = field(init=False, repr=False)
    marker_size_mm: int = field(init=False, repr=False)
    marker_thickness_mm: int = field(init=False, repr=False)

    def __post_init__(self):
        # If the dataclass is frozen, we cannot set the computed values directly (https://stackoverflow.com/a/54119384/400948)
        object.__setattr__(self, 'r_circles_mm', self.dia_circles_mm/2)
        object.__setattr__(self, 'circles_per_row', ((self.target_width_mm - 2*self.min_margin_target_mm - self.dia_circles_mm) // self.dist_circles_mm) + 1)
        object.__setattr__(self, 'circles_per_col', ((self.target_height_mm - 2*self.min_margin_target_mm - self.dia_circles_mm) // self.dist_circles_mm) + 1)
        object.__setattr__(self, 'marker_size_mm', (self.circles_per_square_edge_length - 1) * self.dist_circles_mm + self.dia_circles_mm)
        object.__setattr__(self, 'marker_thickness_mm', (self.circles_per_square_edge_thickness - 1) * self.dist_circles_mm + self.dia_circles_mm)
        self._compute_square_topleft()
        self._compute_skipped_circles()

    def _compute_square_topleft(self):
        gw = self.circles_per_row
        gh = self.circles_per_col
        nx = gw - self.circles_per_square_edge_length
        if nx % 2 == 1:
            print('[Warning] Square marker won''t be centered horizontally.')
        left = nx // 2
        ny = gh - self.circles_per_square_edge_length
        if ny % 2 == 1:
            print('[Warning] Square marker won''t be centered vertically.')
        top = ny // 2
        object.__setattr__(self, 'square_topleft_corner', GridIndex(row=top, col=left))

    def _compute_skipped_circles(self):
        sidx = list()
        for ridx in range(self.circles_per_square_edge_length):
            for cidx in range(self.circles_per_square_edge_length):
                keep = (ridx >= self.circles_per_square_edge_thickness) and\
                        (ridx < self.circles_per_square_edge_length//2) and\
                        (cidx >= self.circles_per_square_edge_thickness) and\
                        (cidx < self.circles_per_square_edge_length - self.circles_per_square_edge_thickness)
                if not keep:
                    sidx.append(GridIndex(row=self.square_topleft_corner.row+ridx,
                                          col=self.square_topleft_corner.col+cidx))
        object.__setattr__(self, 'skipped_circle_indices', sidx)
    
    @property
    def computed_margins(self):
        """
        Returns the actual margins between the target border and
        the first foreground ink/pixels, depending on the pattern's
        specification.
        """
        free_x = self.target_width_mm - (self.circles_per_row - 1) * self.dist_circles_mm - self.dia_circles_mm
        free_y = self.target_height_mm - (self.circles_per_col - 1) * self.dist_circles_mm - self.dia_circles_mm
        return free_x/2, free_y/2

    @property
    def square_rects(self):
        # TODO precompute
        tl = self.square_topleft_corner
        mx, my = self.computed_margins
        offset_x = mx + self.r_circles_mm
        offset_y = my + self.r_circles_mm

        rects = list()
        # Top border
        top = tl.row * self.dist_circles_mm + offset_y - self.r_circles_mm
        left = tl.col * self.dist_circles_mm + offset_x - self.r_circles_mm
        rects.append(Rect(left=left, top=top, width=self.marker_size_mm, height=self.marker_thickness_mm))
        # Left border
        rects.append(Rect(left=left, top=top, width=self.marker_thickness_mm, height=self.marker_size_mm))
        # Right
        x = left + self.marker_size_mm - self.marker_thickness_mm
        rects.append(Rect(left=x, top=top, width=self.marker_thickness_mm, height=self.marker_size_mm))
        # Bottom half
        length_half = self.marker_size_mm / 2
        rects.append(Rect(left=left, top=top+length_half, width=self.marker_size_mm, height=length_half))
        return rects
    
    def skip_circle_idx(self, row, col):
        """
        Returns True if there should be no circle at the current
        grid position (row/col index).
        """
        for skip in self.skipped_circle_indices:
            if row == skip.row and col == skip.col:
                return True
        return False

    def __str__(self):
        mx, my = self.computed_margins
        return f"""[{self.name}]
* Target size: {self.target_width_mm}mm x {self.target_height_mm}mm, margins: {mx}mm, {my}mm
* Circles: d={self.dia_circles_mm}mm, distance={self.dist_circles_mm}mm
* Grid: {self.circles_per_row} x {self.circles_per_col}
* Marker: {self.marker_size_mm}mm x {self.marker_size_mm}mm, border: {self.marker_thickness_mm}mm
* Colors: {self.fg_color} on {self.bg_color}"""
#TODO add remaining specs to __str__

    def export_svg(self, filename):
        print(f'Drawing calibration pattern: {self}')
        h_mm = self.target_height_mm
        w_mm = self.target_width_mm

        # Helper to put fully specified coordinates (in millimeters)
        def _mm(v):
            return f'{v}mm'

        dwg = svgwrite.Drawing(filename=filename, profile='full')
    #, height=f'{h_target_mm}mm', width=f'{w_target_mm}mm', profile='tiny', debug=False)
        # Height/width weren't set properly in the c'tor (my SVGs had 100% instead
        # of the desired dimensions). Thus, set the attributes manually:
        dwg.attribs['height'] = _mm(h_mm)
        dwg.attribs['width'] = _mm(w_mm)

        dwg.defs.add(dwg.style(f".grid {{ stroke: {self.fg_color}; stroke-width:1px; }}"))

        # Background should not be transparent
        dwg.add(dwg.rect(insert=(0, 0), size=(_mm(w_mm), _mm(h_mm)), fill=self.bg_color))

        # Draw circles
        margin_x, margin_y = self.computed_margins
        offset_x = margin_x + self.r_circles_mm
        offset_y = margin_y + self.r_circles_mm

        grid = dwg.add(dwg.g(id='circles'))
        cy_mm = offset_y
        for ridx in range(self.circles_per_col):
            cx_mm = offset_x
            for cidx in range(self.circles_per_row):
                if not self.skip_circle_idx(ridx, cidx):
                    grid.add(dwg.circle(center=(_mm(cx_mm), _mm(cy_mm)),
                        r=_mm(self.r_circles_mm), fill=self.fg_color))
                cx_mm += self.dist_circles_mm
            cy_mm += self.dist_circles_mm
        # Draw Eddie's face
        marker = dwg.add(dwg.g(id='marker'))
        for r in self.square_rects:
            marker.add(dwg.rect(insert=(_mm(r.left), _mm(r.top)),
                                size=(_mm(r.width), _mm(r.height)),
                                class_="grid"))#fill=pspecs.bg_color)
        #FIXME remove
        # # Add a dummy rect to check the printed area
        # dwg.add(dwg.rect(insert=(_mm(pspecs.min_margin_target_mm), _mm(pspecs.min_margin_target_mm)),
        #                  size=(_mm(w_mm - 2*pspecs.min_margin_target_mm), _mm(h_mm - 2*pspecs.min_margin_target_mm)),
        #                  stroke='black', stroke_width='1mm', fill='none'))
        dwg.save()


# eddie_specs_v1 = PatternSpecification('eddie-v1-alu',
#     target_width_mm=300, target_height_mm=400,
#     dia_circles_mm=5, dist_circles_mm=11)

eddie_specs_v1 = PatternSpecificationEddie('eddie-calib-test-a4',
    target_width_mm=210, target_height_mm=297,
    dia_circles_mm=5, dist_circles_mm=11)


def convert_svg_pattern(svg_filename, output_filename):
    """Saves the given SVG file as PDF and PNG."""
    drawing = svg2rlg(svg_filename)
    renderPDF.drawToFile(drawing, f"{output_filename}.pdf")
    renderPM.drawToFile(drawing, f"{output_filename}.png", fmt="PNG")


if __name__ == '__main__':
    pattern_name = 'eddie-v1'
    svg_filename = f'{pattern_name}.svg'
    eddie_specs_v1.export_svg(svg_filename)
    convert_svg_pattern(svg_filename, pattern_name)

