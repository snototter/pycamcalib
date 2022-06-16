import logging
import numpy as np
import toml
from pathlib import Path

from pcc.preproc import filters


_logger = logging.getLogger('pcc.preproc')


class Preprocessor(object):
    """The image preprocessing pipeline.

    This class handles the preprocessing workflow, *i.e.* it holds a list of
    filters which are subsequently applied to a given image.
    """
    def __init__(self):
        self.filters = list()

    def num_filters(self) -> int:
        """Returns the number of filters."""
        return len(self.filters)

    def add_filter(self, filter: filters.FilterBase) -> None:
        """Adds the filter to the preprocessing pipeline."""
        _logger.info(f'Adding filter #{len(self.filters) + 1}: {filter}')
        self.filters.append(filter)

    def set_enabled(self, index: int, enabled: bool) -> None:
        """Enables/disables the filter at the given index.
        
        If index is None or < 0, then the flag will be applied to the whole
        preprocessing pipeline.
        """
        if (index is None) or (index < 0):
            for index in range(len(self.filters)):
                self.filters[index].set_enabled(enabled)
        else:
            self.filters[index].set_enabled(enabled)

    def swap_previous(self, index: int) -> None:
        """Swaps filter at index with filter at index-1.
        
        This method does not implement a parameter sanity check.
        """
        self.filters[index], self.filters[index-1] = self.filters[index-1], self.filters[index]
    
    def swap_next(self, index: int) -> None:
        """Swap operation at index with filter at index+1.
        
        This method does not implement a parameter sanity check.
        """
        self.filters[index], self.filters[index+1] = self.filters[index+1], self.filters[index]

    def remove(self, index: int) -> None:
        """Removes filter at the given index.
        
        This method does not implement a parameter sanity check.
        """
        del self.filters[index]

    def apply(self, image: np.ndarray, num_steps: int = -1) -> np.ndarray:
        """Applies the configured operations subsequently to the given image.

        If ``num_steps < 0``, all operations are applied. Otherwise, only the first
        ``num_steps`` will be applied."""
        if image is None:
            return None
        for step, op in enumerate(self.filters):
            if num_steps >= 0 and step >= num_steps:
                break
            image = op.apply(image)
        return image

    def get_configuration(self) -> dict:
        """Returns a dictionary holding the complete pipeline configuration.
        
        This dictionary can be used to restore the current pipeline via
        :meth:`set_configuration`.

        The nested structure is intended to allow either having a separate
        configuration file for this pipeline or to merge it with other
        configurations.
        """
        return {
            'preproc': {
                'filters': [f.get_configuration() for f in self.filters]
            }
        }

    def save_toml(self, filename):
        """Stores the preprocessing pipeline as TOML configuration."""
        with open(filename, 'w') as fp:
            toml.dump(self.get_configuration(), fp)

    def load_toml(self, filename):
        """Loads the preprocessing pipeline from the given TOML file.

        The filters must be configured as entries of the TOML table
        ``[[preproc.filters]]``, see :meth:`get_configuration`."""
        _logger.info(f'Loading the preprocessing pipeline from `{filename}`')
        config = toml.load(filename)
        self.filters.clear()
        for filter_config in config['preproc']['filters']:
            fname = filter_config['filter']
            f = filters.create_filter(fname)
            f.set_configuration(filter_config)
            self.add_filter(f)
