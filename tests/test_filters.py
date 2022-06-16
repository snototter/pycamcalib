import pytest
import numpy as np
from pcc.preproc import filters


class DummyFilter(filters.FilterBase):
    pass


def base_checks(f):
    """Check that the given filter overrides the required functionality.
    
    For its `apply()`, this helper only tests whether the basic sanity checks
    are implemented. Any real functionality must be tested in the specialized
    test_xxx_filter() function.
    """
    # Dummy simply subclasses FilterBase (without any overrides). This
    # is checked within `test_filter_registration`
    dummy = DummyFilter()

    assert f.filter_name() != filters.FilterBase.filter_name()
    assert f.display_name() != filters.FilterBase.filter_name()
    assert str(f) != str(dummy)
    assert str(f).startswith(f.display_name())
    assert f.__repr__().startswith(f.filter_name())

    # Enable/disable the filter
    for enable in [False, True]:
        f.set_enabled(enable)
        assert f.enabled == enable
    
    # Check if the sanity checks are properly implemented:
    assert f.apply(None) is None
    # If enabled, a filter should act like the identity function:
    f.set_enabled(False)
    assert f.apply(None) is None
    assert f.apply('Foo') == 'Foo'
    
    # Serialize/deserialize --> if a filter is disabled, the
    # flag must be in the output dictionary:
    config = f.get_configuration()
    assert isinstance(config, dict)
    assert 'enabled' in config
    assert not config['enabled']

    restored = filters.create_filter(f.filter_name())
    restored.set_configuration(config)
    assert restored.get_configuration() == config

    # Don't forget to reenable the filter
    f.set_enabled(True)


def test_filter_creation():
    # Invalid filter name must raise an exception
    with pytest.raises(KeyError):
        filters.create_filter('no-such-filter')
    with pytest.raises(KeyError):
        filters.create_filter(None)
    with pytest.raises(KeyError):
        filters.create_filter('')
    
    # Each registered filter must have a default constructor
    for fname in filters.list_filter_names():
        f = filters.create_filter(fname)
        assert isinstance(f, filters.FilterBase)


def test_filter_registration():
    # Only subclasses of FilterBase can be registered as a filter
    with pytest.raises(ValueError):
        filters.register_filter('invalid-class', object)
    with pytest.raises(ValueError):
        filters.register_filter('noop-filter', filters.FilterBase)
    # Register our dummy filter for testing:
    filter_name1 = 'dummy-filter'
    filters.register_filter(filter_name1, DummyFilter)
    # The name should now be taken:
    with pytest.raises(KeyError):
        filters.register_filter(filter_name1, DummyFilter)
    # But we could register it under a different name:
    filter_name2 = 'alias-for-the-dummy-filter'
    filters.register_filter(filter_name2, DummyFilter)

    # Try to create the dummy filter:
    f = filters.create_filter(filter_name1)
    assert isinstance(f, DummyFilter)

    f = filters.create_filter(filter_name2)
    assert isinstance(f, DummyFilter)
    with pytest.raises(NotImplementedError):
        f.apply(None)

    # Check that dummy does not override filter/display name,
    # because we use a dummy instance in the `base_checks` helper
    assert f.filter_name() == filters.FilterBase.filter_name()
    assert f.display_name() == filters.FilterBase.display_name()

    # Clean up the global registry:
    filters.unregister_filter(filter_name1)
    filters.unregister_filter(filter_name2)


def test_interface_compatibility():
    """Test whether each registered filter implements the FilterBase
    interface properly."""
    for fname in filters.list_filter_names():
        f = filters.create_filter(fname)
        base_checks(f)


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


def test_thresholding_filter():
    """Tests the basic functionality of the thresholding filter."""
    f = filters.create_filter('threshold')

    data = np.ones((2, 2, 3), dtype=np.float)
    data[:, :, 1] = 0.8
    data[:, :, 2] = 0.3
    app = f.apply(data)
    assert app.ndim == 2
    L = 0.2989 * data[0, 0, 0] + 0.5870 * data[0, 0, 1] + 0.1140 * data[0, 0, 2]
    assert app == pytest.approx(L)