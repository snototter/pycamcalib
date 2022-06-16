import pytest
import numpy as np
from pcc.preproc import filters


def test_grayscale_filter():
    """Tests the basic functionality of the grayscale conversion filter."""
    f = filters.create_filter('grayscale')

    data = np.ones((2, 2, 3), dtype=np.float)
    data[:, :, 1] = 0.8
    data[:, :, 2] = 0.3
    app = f.apply(data)
    assert app.ndim == 2
    L = 0.2989 * data[0, 0, 0] + 0.5870 * data[0, 0, 1] + 0.1140 * data[0, 0, 2]
    assert app == pytest.approx(L)


def test_gamma_correction_filter():
    """Tests the basic functionality of the gamma correction filter."""
    f = filters.create_filter('gamma')
    f.set_gamma(0.2)
    assert f.gamma == pytest.approx(0.2)
    f.set_gamma(1.7)
    assert f.gamma == pytest.approx(1.7)

    # Check that de-/serialization restores the gamma parameter
    d = f.get_configuration()
    assert 'gamma' in d
    assert d['gamma'] == pytest.approx(1.7)

    restored = filters.create_filter('gamma')
    restored.set_configuration(d)
    assert restored.gamma == pytest.approx(1.7)

    # Apply filter on test data (invalid inputs have already been
    # checked at test_interface_compatibility)
    data = np.random.randint(0, 256, size=(3, 3)).astype(np.uint8)
    app = f.apply(data)
    assert isinstance(app, np.ndarray)
    # Gamma = 1 is the identity transformation:
    f.set_gamma(1.0)
    app = f.apply(data)
    assert np.allclose(data, app)
