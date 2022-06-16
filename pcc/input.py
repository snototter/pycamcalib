import os
from pathlib import Path
import numpy as np
from vito import imutils


class CalibrationImage(object):
    """Encapsulates the image data, filename, etc."""
    def __init__(self,
            id: int,
            image: np.ndarray,
            filename: str
        ):
        self.id = id
        self.image = image
        self.filename = filename
    
    def __eq__(self, other) -> bool:
        if isinstance(other, CalibrationImage):
            return (self.id == other.id)\
                and np.allclose(self.image, other.image)\
                and (self.filename == other.filename)
        else:
            return False

    def __str__(self) -> str:
        shapestr = 'None' if self.image is None else 'x'.join([str(s) for s in self.image.shape])
        tp = None if self.image is None else self.image.dtype
        return f'#{self.id}: {shapestr}, {tp}, `{self.filename}`'


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
    def __init__(self,
            folder: Path,
            preload_images: bool = False
        ):
        if not isinstance(folder, Path):
            folder = Path(folder)
        if not folder.exists():
            raise DirectoryNotFoundError(f'Image folder `{folder}` does not exist!')
        if not folder.is_dir():
            raise DirectoryNotFoundError(f'Path `{folder}` is not a directory!')
        self.folder = folder
        self.files = sorted([f.name for f in folder.iterdir() if _is_image_filename(f)])
        self.current_idx = 0
        if len(self.files) == 0:
            raise NoImageDirectoryError(f'No image files found in folder `{folder}`!')
        # Load all images into memory if requested
        self.images = [None] * len(self.files)
        if preload_images:
            self.load_all()

    def is_available(self) -> bool:
        """Returns True if there is a file not yet accessed via next_image()."""
        return 0 <= self.current_idx < len(self.files)

    def next_image(self) -> CalibrationImage:
        """Loads the next image."""
        if not self.is_available():
            return None
        img = self.load_index(self.current_idx)
        self.current_idx += 1
        return img

    def num_images(self) -> int:
        return len(self.files)

    def filenames(self) -> list:
        """Returns a list of all (basename) image names within the folder."""
        return self.files

    def load_all(self) -> None:
        """Loads all images into memory."""
        for idx in range(self.num_images()):
            self.load_index(idx)
    
    def load_index(self, index: int) -> CalibrationImage:
        if self.images[index] is None:
            fn = self.folder / self.files[index]
            self.images[index] = imutils.imread(fn)
        return CalibrationImage(index, self.images[index], self.files[index])

    def __getitem__(self, index: int) -> CalibrationImage:
        return self.load_index(index)

    def __iter__(self):
        return self

    def __next__(self) -> CalibrationImage:
        img = self.next_image()
        if img is None:
            raise StopIteration
        return img
