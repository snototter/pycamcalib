import logging
import os
import io
import numpy as np
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from vito import pyutils, imutils


def export_svgwrite_drawing(dwg: svgwrite.Drawing, filename: str):
    dwg.saveas(filename, pretty=True)


def svgwrite2image(dwg: svgwrite.Drawing) -> np.ndarray:
    """Converts the given SVG drawing to an in-memory image file (NumPy ndarray)."""
    # Load the rendered SVG into a StringIO
    svg_sio = io.StringIO(dwg.tostring())
    # Render it to PNG in-memory
    dwg_input = svg2rlg(svg_sio)
    img_mem_file = io.BytesIO()
    renderPM.drawToFile(dwg_input, img_mem_file, fmt="PNG")  # NEVER EVER SET DPI! doesn't scale properly as of 2021-09
    return imutils.memory_file2ndarray(img_mem_file)


def export_board(board_specification, output_basename: str=None, output_folder: str='.',
                 export_pdf: bool=True, export_png: bool=True, export_svg: bool=True,
                 prevent_overwrite: bool=True):
    """Saves the given calibration pattern to disk.
    
:board_specification: Contains the calibration board specification.

:output_basename:   If None, the board_specification's slugified identifier
                    will be used. Otherwise, the file(s) will be stored
                    as <basename>.<extension>

:output_folder:     Folder where to save the files

:export_pdf, export_png, export_svg: Flags to select the desired output
                    format(s).

:prevent_overwrite: If True and the output file(s) already exist(s), a
                    FileExistsError will be raised
    """
    if all([f is False for f in [export_pdf, export_png, export_svg]]):
        raise ValueError('You must enable at least one export format!')

    if output_basename is None:
        output_basename = pyutils.slugify(board_specification.name)

    # Prevent overwrite if requested:
    fn_svg = os.path.join(output_folder, output_basename + '.svg')
    fn_pdf = os.path.join(output_folder, output_basename + '.pdf')
    fn_png = os.path.join(output_folder, output_basename + '.png')
    if prevent_overwrite:
        logging.info(f'Checking existing files at output location to prevent overwriting.')
        to_check = list()
        if export_pdf:
            to_check.append((fn_pdf, 'PDF'))
        if export_png:
            to_check.append((fn_png, 'PNG'))
        if export_svg:
            to_check.append((fn_svg, 'SVG'))
        for fn, tp in to_check:
            if os.path.exists(fn):
                raise FileExistsError(f'{tp} file already exists: {fn}')

    if not os.path.exists(output_folder):
        logging.info(f'Creating output folder: {output_folder}')
        os.makedirs(output_folder)

    svg_dwg = board_specification.svg()

    if export_svg:
        logging.info(f'Exporting {board_specification.name} to {fn_svg}')
        export_svgwrite_drawing(svg_dwg, fn_svg)

    if export_pdf or export_png:
        # Parse svgwrite.Drawing into svglib
        drawing = svg2rlg(io.StringIO(svg_dwg.tostring()))
        if export_pdf:
            logging.info(f'Exporting {board_specification.name} to {fn_pdf}')
            renderPDF.drawToFile(drawing, fn_pdf)
        if export_png:
            #TODO increasing dpi doesn't work: https://github.com/deeplook/svglib/issues/207 
            # Future ideas: use inkscape (adds an external requirement, but their PNG export works
            # perfectly)
            logging.info(f'Exporting {board_specification.name} to {fn_png}')
            renderPM.drawToFile(drawing, fn_png, fmt="PNG")


def overlay_pattern_specification(dwg: svgwrite.Drawing, text_line1: str, text_line2: str,
                                  board_height_mm: float, available_space_mm: float, offset_left_mm: float,
                                  font_size_mm: int = 4, line_padding_mm: int = 1,
                                  overlay_color: str = 'rgb(120, 120, 120)'):
    """Overlays the pattern specification strings onto the SVG drawing.
    Depending on the available free space at the bottom of the board this will:
    * Place both text_line arguments below each other, or
    * Concatenate the text_line arguments and place them as a single line, or
    * Don't add any text and log a warning instead.
    """
    # Check if 2 lines fit within the available space. If not, we try only a single line of text:
    text_height_mm = 2 * (font_size_mm + line_padding_mm)
    single_line = text_height_mm > available_space_mm
    if single_line:
        text_height_mm = font_size_mm + line_padding_mm
    
    if available_space_mm < text_height_mm:
        logging.warning(f'Cannot overlay specification. Available free space {available_space_mm}mm is too small (requiring at least {text_height_mm} mm).')
    else:
        def _mm(v):
            return f'{v}mm'

        top = min(board_height_mm - available_space_mm, board_height_mm - text_height_mm) + font_size_mm
        overlay = dwg.add(dwg.g(style=f"font-size:{_mm(font_size_mm)};font-family:monospace;stroke:none;fill:{overlay_color};"))
        if single_line:
            overlay.add(dwg.text(f'{text_line1} {text_line2}',
                                 insert=(_mm(offset_left_mm), _mm(top))))
        else:
            overlay.add(dwg.text(text_line1,
                                 insert=(_mm(offset_left_mm), _mm(top))))
            top += font_size_mm + line_padding_mm
            overlay.add(dwg.text(text_line2,
                                 insert=(_mm(offset_left_mm), _mm(top))))
