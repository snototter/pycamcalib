# Development guidelines

* Test and report line coverage:
  ```bash
  pip install pytest-cov
  pytest --cov-config=tests/.coveragerc --cov=pcc/

  # Report missed lines:
  pytest --cov-config=tests/.coveragerc --cov=pcc/ --cov-report term-missing
  ```
* Linting:
  ```bash
  flake8 --max-line-length=127 tests/test_imutils.py
  ```
* Tasks before release:
  * Update `tests/.coveragerc` (non-refactored code is excluded)!
  * Set up automated packaging, tests, linting, travis, ...
* Stick to the [Google style guide](https://google.github.io/styleguide/pyguide.html) as closely as possible

