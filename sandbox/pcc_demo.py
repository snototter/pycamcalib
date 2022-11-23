import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')))

from pcc import patterns
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    eddie_a4 = patterns.eddie.PatternSpecificationEddie(
        'Eddie Test Pattern A4', target_width_mm=210, target_height_mm=297,
        dia_circles_mm=5, dist_circles_mm=11)
    patterns.export_board(
        eddie_a4, output_folder='test-folder', output_basename=None,
        export_pdf=True, export_png=False, prevent_overwrite=False)
