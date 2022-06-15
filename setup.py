import setuptools
import os

# Load version string
loaded_vars = dict()
with open(os.path.join(os.path.dirname(__file__), 'pcc', 'version.py')) as fv:
    exec(fv.read(), loaded_vars)

setuptools.setup(
    name="pcc",
    version=loaded_vars['__version__'],
    author="snototter",
    author_email="snototter@users.noreply.github.com",
    description="Intrinsic camera calibration toolbox.",
    url="https://github.com/snototter/pycamcalib",
    packages=setuptools.find_packages(),
    install_requires=[
        'dataclasses',  # introduced to standard packages in py3.7
        'numpy',
        'svglib',
        'svgwrite',
        'opencv-python-headless',
        'pyside2',
        'qimage2ndarray',
        'toml',
        'vito'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
