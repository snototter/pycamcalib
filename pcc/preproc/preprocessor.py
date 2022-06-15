import cv2
import logging
import numpy as np
import toml
import typing
from vito import imutils

from pcc.preproc import filters

#TODO implement additional ops:
# * threshold variants, see e.g. https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
# * maybe add naive contrast adjustment I*alpha + beta https://towardsdatascience.com/exploring-image-processing-techniques-opencv-4860006a243


_logger = logging.getLogger('pcc.preproc')


class Preprocessor(object):
    """Implements the preprocessing pipeline.

TODO This is the main "workflow" class, i.e. it holds a list of operations which are
applied subsequently to a given image via 'process()'.
    """
    def __init__(self):
        self.filters = list()
        self.enabled = True

    def num_operations(self) -> int:
        return len(self.operations)

    def add_filter(self, filter: filters.FilterBase) -> None:
        _logger.info(f'Adding filter #{len(self.filters)}: {filter}')
        self.filters.append(filter)

    def set_enabled(self, index: int, enabled: bool) -> None:
        """Enable/disable the operation at the given index.TODO if none....."""
        if index is None:
            self.enabled = enabled
        else:
            self.operations[index].set_enabled(enabled)

    def swap_previous(self, index: int) -> None:
        """Swap operation at index with operation at index-1. TODO out of range check!?"""
        self.operations[index], self.operations[index-1] = self.operations[index-1], self.operations[index]
    
    def swap_next(self, index: int) -> None:
        """Swap operation at index with operation at index+1."""
        self.operations[index], self.operations[index+1] = self.operations[index+1], self.operations[index]

    def remove(self, index: int) -> None:
        """Remove operation at index"""
        del self.operations[index]

    def apply(self, image: np.ndarray, num_steps: int = -1) -> np.ndarray: #TODO image type!
        """Applies the configured operations subsequently to the given image.
        If num_steps <= 0, all operations are applied. Otherwise, only the first
        num_steps will be applied."""
        if image is None:
            return None
        if not self.enabled:
            return image
        for step, op in enumerate(self.operations):
            if num_steps >= 0 and step >= num_steps:
                break
            image = op.apply(image)
        return image

    def get_configuration(self) -> dict:
        """Returns a dictionary which can be used to restore the current state
        of this preprocessing pipeline.
        Nested dict structure is intended to allow a) having a separate configuration
        file for this pipeline and b) merge it with other (sub)module configurations.
        """
        return {
            'preproc': {
                'filters': [f.get_configuration() for f in self.filters],
                'enabled': self.enabled #TODO
            }
        }

    def save_toml(self, filename: str) -> None:
        """Stores the current state of this preprocessing pipeline to disk."""
        with open(filename, 'w') as fp:
            toml.dump(self.get_configuration(), fp)

    def load_toml(self, filename: str) -> None:
        """Loads the preprocessing pipeline from the given TOML file.

        The filters must be configured as entries of the TOML table
        ``[[preproc.filters]]``."""
        _logger.info(f'Loading the preprocessing pipeline from `{filename}`')
 #       try:
        config = toml.load(filename)
        self.filters.clear()
        for filter_config in config['preproc']['filters']:
            print(f'TODO RM {filter_config}')
            fname = filter_config['filter']
            f = filters.create_filter(fname)
            f.set_configuration(filter_config)
            self.add_filter(f)
#        except KeyError as e:
#            raise ConfigurationError(f"Invalid/missing configuration key '{str(e)}'") from None
#        except FileNotFoundError:
#            raise ConfigurationError(f"Configuration file not found: '{filename}'") from None
#        except (toml.TomlDecodeError, TypeError) as e:
#            raise ValueError(f"Invalid TOML configuration: {e}") from None

