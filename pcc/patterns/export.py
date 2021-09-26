import logging
import os
import io
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from vito import pyutils


def _export_svgwrite_drawing(dwg: svgwrite.Drawing, filename: str):
    dwg.saveas(filename, pretty=True)


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
        _export_svgwrite_drawing(svg_dwg, fn_svg)

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
