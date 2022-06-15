from typing import Any

class FilterBase(object):
    """Base class for all preprocessing filters.
    
    Basic requirements for each filter:
    * it must OVERRIDE TODO a unique class-wide `name` attribute
    * it must OVERRIDE TODO display_name; filter_name; + override __str__ !
a class-wide 'display' attribute (speaking name for class selection before instantiation)
    * it must implement `description()`-> str' to return a brief but nicely
      formatted name (used for display within the UI). You can include a summary
      of its configuration, e.g. "Threshold (t=128)"
    * it must implement 'apply(np.ndarray) -> np.ndarray'
    * it must be registered TODO(!!) within 'AVAILABLE_PREPROCESSOR_OPERATIONS'

TODO doc!
    Life-cycle of a filter:
    * Construction via default c'tor. This must initialize all parameters to
      sane default values.
    * Configuration via :meth:`configure`. This must set all parameters from
      the input dictionary.
    * it must override 'configure(dict) -> None' to load parameters from a dictionary
    * it must override 'freeze() -> dict' to store all parameters in a dictionary
    * it should override '__repr__() -> str'
    """

    @staticmethod
    def filter_name() -> str:
        """Returns the unique name used in configuration files.

        Best practices:
        * Use only lower-case names.
        * Only include alpha-numeric letters + hyphens, i.e. avoid special
          characters, whitespace.
        """
        return "filter-base"

    @staticmethod
    def display_name() -> str:
        """TODO"""
        return "Basic Filter"

    def __init__(self):
        self.enabled = True


    def set_enabled(self, enabled) -> None:
        """Enables/disables this filter instance."""
        self.enabled = enabled

    def set_configuration(self, config: dict) -> None:
        """Deserializes this filter's configuration from the given
        dictionary."""
        # By design, the 'enabled' field is optional in our TOML specification.
        if 'enabled' in config:
            self.enabled = config['enabled']

    def get_configuration(self) -> dict:
        """Serializes this filter's configuration as a dictionary."""
        d = {'filter': type(self).filter_name()}
        # Operations are enabled by default. Thus, we omit this field
        # for TOML serialization
        if not self.enabled:
            d['enabled'] = self.enabled
        return d

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return type(self).display_name()


# List of all available preprocessing operations.
# Since we want to provide a custom ordering of the operations in the UI, this
# list cannot be retrieved automatically (e.g. via inspect)
AVAILABLE_FILTERS = dict()

def register_filter(filter_name: str, cls: Any) -> None:
    global AVAILABLE_FILTERS
#TODO check if name unique
    if filter_name in AVAILABLE_FILTERS:
        raise KeyError(f'Filter name `{filter_name}` has already been registered.')
    AVAILABLE_FILTERS[filter_name] = cls


def create_filter(filter_name: str) -> FilterBase:
    if filter_name not in AVAILABLE_FILTERS:
        raise KeyError(f'Filter with name `{filter_name}` has not been registered!')
    return AVAILABLE_FILTERS[filter_name]()

