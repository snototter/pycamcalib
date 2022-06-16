import os
from pathlib import Path

TESTS_DIR = Path(os.path.dirname(__file__)).absolute()

EXAMPLE_DATA_DIR = (TESTS_DIR / '..' / 'examples' / 'data').absolute()

TEST_DATA_DIR = EXAMPLE_DATA_DIR / 'tests'
