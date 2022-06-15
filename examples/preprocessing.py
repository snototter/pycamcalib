import os
import time
from vito import imvis, imutils

from pcc.preproc import Preprocessor

def demo_preproc(cfg_file, img_file):
    preproc = Preprocessor()
    preproc.load_toml(cfg_file)

    print(preproc.get_configuration())

    for f in preproc.filters:
        print(f'Filter: {f}')

    img = imutils.imread(img_file)
    for step in range(preproc.num_filters()):
        img_pp = preproc.apply(img, step)
        imvis.imshow(img_pp, title=f'Preprocessed Image #{step}')
        time.sleep(0.2)


if __name__ == '__main__':
    cfg_file = os.path.join(os.path.dirname(__file__), 'data', 'cfg-preproc.toml')
    img_file = os.path.join(os.path.dirname(__file__), 'data', 'flamingo.jpg')
    demo_preproc(cfg_file, img_file)
