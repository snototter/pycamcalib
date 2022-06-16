import argparse
from pathlib import Path

from pcc import input, preproc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_directory', action='store', type=Path,
        help="Path to the folder containing the calibration images."
    )
    parser.add_argument(
        'pattern_config', action='store', type=Path,
        help="The TOML file containing the pattern specification."
    )
    parser.add_argument(
        '--preproc-config', '--preproc', dest='preproc_config', action='store',
        type=Path, default=None,
        help='The TOML file containing the configuration of the preprocessing pipeline.')
    
    return parser.parse_args()


def calibrate_mono_cli():
    args = parse_args()
    image_source = input.ImageSource(args.image_directory)
    preprocessor = preproc.Preprocessor()
    if args.preproc_config is not None:
        preprocessor.load_toml(args.preproc_config) 

    for calib_img in image_source:
        print(f'Processing {calib_img}')


if __name__ == '__main__':
    calibrate_mono_cli()
