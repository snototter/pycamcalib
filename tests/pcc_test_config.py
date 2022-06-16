import os
from pathlib import Path

TESTS_DIR = Path(os.path.dirname(__file__)).absolute()

DATA_DIR = (TESTS_DIR / '..' / 'examples' / 'data').absolute()

TEST_DATA_DIR = DATA_DIR / 'tests'
