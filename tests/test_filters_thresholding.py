import pytest
import numpy as np
from pcc.preproc import filters


def test_thresholding_filter():
    """Tests the basic functionality of the thresholding filter."""
    f = filters.create_filter('threshold')

    # Test all setters
    f.set_threshold_value(17)
    assert f.threshold_value == 17

    f.set_max_value(100)
    assert f.max_value == 100

    with pytest.raises(ValueError):
        f.set_threshold_type('foo')
    for tp in filters.Thresholding.THRESHOLD_TYPES:
        f.set_threshold_type(tp)
    f.set_threshold_type('binary')

    with pytest.raises(ValueError):
        f.set_method('bar')
    for m in filters.Thresholding.METHODS:
        f.set_method(m)

    # Serialization
    f.set_method('otsu')
    d = f.get_configuration()
    assert 'threshold_value' not in d
    assert d['max_value'] == 100
    assert d['method'] == 'otsu'
    assert d['type'] == f._type2str(f.threshold_type)

    f.set_method('global')
    d = f.get_configuration()
    assert d['threshold_value'] == 17

    restore = filters.create_filter(f.filter_name())
    restore.set_configuration(d)
    assert restore.get_configuration() == d

    # Standard type/method to string conversion is tested via
    # get_configuration(). We need to check invalid inputs, though:
    with pytest.raises(ValueError):
        f._type2str(99)
    with pytest.raises(ValueError):
        f._method2str(-5)

    # Apply on dummy data
    data = np.ones((2, 2, 3), dtype=np.uint8)
    data[:, :, 1] = 255
    data[:, :, 2] = 128
    app = f.apply(data)
    assert np.all(app[:, :, 0] == 0)
    assert np.all(app[:, :, 1:] == 100)

    f.set_method('triangle')
    app = f.apply(data)
    assert app.ndim == 2
    assert np.all(app[:] == 100)


def test_adaptive_thresholding_filter():
    """Tests the basic functionality of the adaptive thresholding filter."""
    f = filters.create_filter('adaptive-threshold')

    # Test all setters
    f.set_max_value(200)
    assert f.max_value == 200

    with pytest.raises(ValueError):
        f.set_threshold_type('foo')
    for tp in filters.AdaptiveThresholding.THRESHOLD_TYPES:
        f.set_threshold_type(tp)
    f.set_threshold_type('binary')

    with pytest.raises(ValueError):
        f.set_method('bar')
    for m in filters.AdaptiveThresholding.METHODS:
        f.set_method(m)
    f.set_method('mean')

    with pytest.raises(ValueError):  # Block size must be odd
        f.set_block_size(4)
    f.set_block_size(7)
    assert f.block_size == 7
    
    f.set_C(3)
    assert f.C == 3

    # Serialization
    d = f.get_configuration()
    restore = filters.create_filter(f.filter_name())
    restore.set_configuration(d)
    assert restore.get_configuration() == d
    assert restore.max_value == f.max_value
    assert restore.threshold_type == f.threshold_type
    assert restore.method == f.method
    assert restore.block_size == f.block_size
    assert restore.C == f.C

    # Standard type/method to string conversion is tested via
    # get_configuration(). We need to check invalid inputs, though:
    with pytest.raises(ValueError):
        f._type2str(42)
    with pytest.raises(ValueError):
        f._method2str(23)

    # Apply on dummy data
    data = np.ones((2, 2, 3), dtype=np.uint8)
    data[:, :, 1] = 255
    data[:, :, 2] = 128
    app = f.apply(data)
    assert app.ndim == 2
    assert np.all(app[:] == 200)
