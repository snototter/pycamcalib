import os
import numpy as np
import typing
from vito import imutils


def _is_image_filename(f):
    """Returns True if the given filename has a supported image format extension."""
    # We only test for the most common formats, supported by OpenCV and/or Pillow
    img_extensions = ['.bmp', '.jpeg', '.jpg', '.png', '.ppm', '.tif', '.webp']
    _, ext = os.path.splitext(f)
    return ext.lower() in img_extensions


class ImageDirectorySource(object):
    """Allows iterating all images of a local directory in sorted order."""
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if _is_image_filename(f)])
        self.idx = 0

    def is_available(self) -> bool:
        return 0 <= self.idx < len(self.files)  # Python allows chained comparisons

    def next(self) -> typing.Tuple[np.ndarray, str]:
        if not self.is_available():
            return None, None
        fn = os.path.join(self.folder, self.files[self.idx])
        img = imutils.imread(fn)
        self.idx += 1
        return img, fn
