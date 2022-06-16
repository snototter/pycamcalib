import pytest
import numpy as np
from pcc.preproc import filters


def test_histeq_filter():
    """Tests the basic functionality of the histogram equalization filter."""
    f = filters.create_filter('histeq')

    # Apply on dummy data
    data = np.ones((2, 2, 3), dtype=np.uint8)
    data[:, :, 1] = 255
    data[:, :, 2] = 128
    data[0, 0, :] = 0
    app = f.apply(data)
    assert app.ndim == 3
    assert np.array_equal(app[0, 0, :], (0, 0, 0))
    assert np.array_equal(app[0, 1, :], (91, 255, 218))
    assert np.array_equal(app[1, 0, :], (91, 255, 218))
    assert np.array_equal(app[1, 1, :], (91, 255, 218))
    
    data = data[:, :, 0].reshape((2, 2))
    app = f.apply(data)
    assert app.ndim == 2
    assert app[0, 0] == 0
    assert app[0, 1] == 255
    assert app[1, 0] == 255
    assert app[1, 1] == 255


def test_clahe_filter():
    """Tests the basic functionality of the CLAHE filter."""
    f = filters.create_filter('clahe')

    # Apply on dummy data
    data = np.ones((2, 2, 3), dtype=np.uint8)
    data[:, :, 1] = 255
    data[:, :, 2] = 128
    data[0, 0, :] = 0
    app = f.apply(data)
    assert app.ndim == 3
    assert np.array_equal(app[0, 0, :], (255, 255, 255))
    assert np.array_equal(app[0, 1, :], (91, 255, 218))
    assert np.array_equal(app[1, 0, :], (91, 255, 218))
    assert np.array_equal(app[1, 1, :], (91, 255, 218))
    
    data = data[:, :, 0].reshape((2, 2))
    app = f.apply(data)
    assert app.ndim == 2
    assert np.all(app[:] == 255)
