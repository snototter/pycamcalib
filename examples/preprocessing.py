import os
from vito import imvis

from pcc.preproc import Preprocessor

def demo_preproc(cfg_file):
    preproc = Preprocessor()
    preproc.load_toml(cfg_file)

    print(preproc.get_configuration())


if __name__ == '__main__':
    cfg_file = os.path.join(os.path.dirname(__file__), 'data', 'cfg-preproc.toml')
    demo_preproc(cfg_file)

