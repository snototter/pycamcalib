import logging
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from vito import pyutils


def export_pattern(pattern_spec, output_folder, output_basename=None,
                   export_pdf=True, export_png=True, prevent_overwrite=True):
    """Saves the given pattern as SVG and optionally PDF and/or PNG.
    
    :pattern_spec:    The PatternSpecification... object

    :output_folder:   Folder where to save the files

    :output_basename: If None, the pattern_spec's slugified identifier
                      will be used. Otherwise, the files will be stored
                      as <basename>.<extension>
    
    TODO doc these flags:
    :export_pdf:
    :export_png:
    :prevent_overwrite:
    """
    if output_basename is None:
        output_basename = pyutils.slugify(pattern_spec.name)
    fn_svg = os.path.join(output_folder, output_basename + '.svg')
    fn_pdf = os.path.join(output_folder, output_basename + '.pdf')
    fn_png = os.path.join(output_folder, output_basename + '.png')
    # Prevent overwrite if requested:
    if prevent_overwrite:
        for fn, tp in [(fn_svg, 'SVG'), (fn_pdf, 'PDF'), (fn_png, 'PNG')]:
            if os.path.exists(fn):
                raise FileExistsError(f'{tp} file already exists: {fn}')
    if not os.path.exists(output_folder):
        logging.info(f'Creating output folder: {output_folder}')
        os.makedirs(output_folder)
    # First, let the pattern export itself to svg...
    pattern_spec.export_svg(fn_svg)
    logging.info(f'Exported {pattern_spec.name} to {fn_svg}')
    # ... then export to png/pdf
    if export_pdf or export_png:
        drawing = svg2rlg(fn_svg)
        if export_pdf:
            renderPDF.drawToFile(drawing, fn_pdf)
            logging.info(f'Exported {pattern_spec.name} to {fn_pdf}')
        if export_png:
            #FIXME increasing dpi doesn't work: https://github.com/deeplook/svglib/issues/207 
            # Maybe switch to inkscape
            renderPM.drawToFile(drawing, fn_png, fmt="PNG")
            logging.info(f'Exported {pattern_spec.name} to {fn_png}')
