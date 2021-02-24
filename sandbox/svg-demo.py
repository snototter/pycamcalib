import svgwrite
import svglib

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

from dataclasses import dataclass

@dataclass
class PatternSpecification:
    name: str
    width_mm: int
    height_mm: int

eddie_specs_v1 = PatternSpecification('eddie-v1', 100, 210)


def make_svg_pattern(pattern_specs, filename, bg_color='white'):
    """
    :pattern_specs: the PatternSpecification to be drawn
    :bg_color: color used to fill the background, specify via:
               * named colors: white, red, orange, ...
               * hex colors: #ff9e2c
               * rgb colors: rgb(255, 128, 44)
    """
    h_mm = pattern_specs.height_mm
    w_mm = pattern_specs.width_mm

    dwg = svgwrite.Drawing(filename=filename, profile='full')
#, height=f'{h_target_mm}mm',
#                           width=f'{w_target_mm}mm', profile='tiny', debug=False)
    # Height/width weren't set properly in the c'tor (my SVGs had 100% instead
    # of the desired dimensions). Thus, set the attributes manually:
    dwg.attribs['height'] = f'{h_mm}mm'
    dwg.attribs['width'] = f'{w_mm}mm'

    dwg.defs.add(dwg.style(".grid { stroke: rgb(0,0,0); stroke-width:1px; }"))

    # Background should not be transparent
    dwg.add(dwg.rect(insert=(0, 0), size=(f'{w_mm}mm', f'{h_mm}mm'), fill=bg_color))

    # Horizontal lines
    grid = dwg.add(dwg.g(id='hlines'))
    for y_mm in range(0, h_mm+1, 5):
        grid.add(dwg.line(start=('0mm', f'{y_mm}mm'), end=(f'{w_mm}mm', f'{y_mm}mm'),
                          class_='grid'))
    dwg.save()


def render_pattern(svg_filename, output_filename):
    drawing = svg2rlg(svg_filename)
    renderPDF.drawToFile(drawing, f"{output_filename}.pdf")
    renderPM.drawToFile(drawing, f"{output_filename}.png", fmt="PNG")


if __name__ == '__main__':
    pattern_name = 'eddie-v1'
    svg_filename = f'{pattern_name}.svg'
#    make_svg_pattern(svg_filename, bg_color='gray')
    make_svg_pattern(eddie_specs_v1, svg_filename, bg_color='rgb(128,0,0)')
    render_pattern(svg_filename, pattern_name)

