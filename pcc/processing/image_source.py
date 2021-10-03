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


class DirectoryNotFoundError(Exception):
    """Wrong directory path."""
    pass


class NoImageDirectoryError(Exception):
    """Directory contains no (supported) image files."""
    pass


class ImageSource(object):
    """Allows iterating all images of a local directory in sorted order."""
    def __init__(self, folder, preload_images=False):
        if not os.path.exists(folder):
            raise DirectoryNotFoundError(f"No such folder '{folder}'")
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if _is_image_filename(f)])
        self.idx = 0
        if len(self.files) == 0:
            raise NoImageDirectoryError(f"No image files in folder '{folder}'")
        # Load all images into memory if requested
        self.images = None if not preload_images else self.load_all()

    def is_available(self) -> bool:
        """Returns True if there is a file not yet accessed via next()."""
        return 0 <= self.idx < len(self.files)  # Python allows chained comparisons

    def next(self) -> typing.Tuple[np.ndarray, str]:
        """Loads the next image and additionally returns the (full) filename."""
        if not self.is_available():
            return None, None
        fn = os.path.join(self.folder, self.files[self.idx])
        img = imutils.imread(fn)
        self.idx += 1
        return img, fn

    def num_images(self):
        return len(self.files)

    def filenames(self):
        """Returns a list of all (basename) image names within the folder."""
        return self.files

    def load_all(self):
        return [imutils.imread(os.path.join(self.folder, self.files[i]))
                for i in range(len(self.files))]

    def __getitem__(self, index: int):
        if self.images is None:
            self.images = self.load_all()
        return self.images[index]
