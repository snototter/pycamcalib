import setuptools

# # Load version string
# loaded_vars = dict()
# with open('helheimr_bot/version.py') as fv:
#     exec(fv.read(), loaded_vars)

setuptools.setup(
    name="pcc",
    version='0.1', #loaded_vars['__version__'],
    author="snototter",
    author_email="snototter@users.noreply.github.com",
    description="Intrinsic camera calibration utility.",
    url="https://github.com/snototter/pycamcalib",
    packages=setuptools.find_packages(),
    install_requires=[
        'vito', 'wheel', 'numpy', 'svglib', 'svgwrite', 'dataclasses',
        'opencv-python', 'pyside2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
