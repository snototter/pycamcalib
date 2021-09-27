import os
from vito import imutils


def _is_image_filename(f):
    _, ext = os.path.splitext(f)
    return ext.lower() in ['.bmp', '.jpeg', '.jpg', '.png', '.ppm', '.tif', '.webp']


class ImageDirectorySource(object):
    """Allows iterating all images of a local directory in sorted order."""
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if _is_image_filename(f)])
        self.idx = 0

    def is_available(self) -> bool:
        return 0 <= self.idx < len(self.files)  # Python allows chained comparisons
    
    def next(self):
        if not self.is_available():
            return None
        img = imutils.imread(os.path.join(self.folder, self.files[self.idx]))
        self.idx += 1
        return img
        