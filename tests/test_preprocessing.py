import pytest
import pytest
from tests.pcc_test_config import TEST_DATA_DIR
from pcc.preproc import Preprocessor, filters

def test_preproc_pipeline():
    p = Preprocessor()
    p.load_toml(TEST_DATA_DIR / 'test-cfg-preproc.toml')

    assert p.num_filters() == len(filters.list_filter_names())

    for f in p.filters:
        assert f.enabled

#TODO test swapping filters
#TODO test applying filters
#TODO test en/disabling filters