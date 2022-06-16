# Development guidelines

* Tasks before release:
  * Update `tests/.coveragerc` (non-refactored code is excluded)!
  * Set up automated packaging, tests, linting, travis, ...
* Stick to the [Google style guide](https://google.github.io/styleguide/pyguide.html) as closely as possible
* Test policy: 
  * Users of pcc should be able to read the docs/documentation and correctly invoke the methods (if they really want to dig that deep into the package)
  * Focus on interface usage & pipeline/workflows. For example, for a Filter we don't need to test the correctness of the underlying algorithms (unless implement specifically for pcc). But we should test whether the FilterBase interface is implemented correctly, and the user would be able to configure such a filter.
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

