import numpy as np
import pytest
from pathlib import Path
from tests.pcc_test_config import EXAMPLE_DATA_DIR
from pcc.input import CalibrationImage, DirectoryNotFoundError, ImageSource, NoImageDirectoryError


def test_invalid_sources():
    with pytest.raises(DirectoryNotFoundError):
        ImageSource('foo-dir')
    with pytest.raises(DirectoryNotFoundError):
        ImageSource(__file__)
    with pytest.raises(NoImageDirectoryError):
        ImageSource(Path(__file__).parent)

def test_image_source():
    # No preloading
    img_source = ImageSource(EXAMPLE_DATA_DIR)

    assert img_source.num_images() == 1
    for idx in range(img_source.num_images()):
        assert img_source.images[idx] is None
    
    assert len(img_source.filenames()) == 1
    assert img_source.filenames()[0] == 'flamingo.jpg'

    assert img_source.is_available()
    index = 0
    for img in img_source:
        assert img is not None
        assert isinstance(img, CalibrationImage)
        print(img)
        assert img == img_source[index]
        assert img != 'foo'
        index += 1
    
    # With preloading
    img_source = ImageSource(
        EXAMPLE_DATA_DIR, preload_images = True)

    assert img_source.num_images() == 1
    for idx in range(img_source.num_images()):
        assert img_source.images[idx] is not None
        assert isinstance(img_source.images[idx], np.ndarray)
    assert img_source.is_available()
