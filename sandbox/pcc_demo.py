import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')))

from pcc import patterns
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    patterns.export_pattern(patterns.eddie.eddie_specs_v1, 'test-folder', None,
                          export_pdf=True, export_png=False,
                          prevent_overwrite=False)
