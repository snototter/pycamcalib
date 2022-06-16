from typing import Any
import numpy as np


class FilterBase(object):
    """Base class for all preprocessing filters.
    
    Basic requirements for each filter:
    * If you implement your own preprocessing filter, it's obviously a good idea
      to check the existing filters first on how it could/should be done ;-)
    * The filter must override :meth:`filter_name`, :meth:`display_name`,
      :meth:`apply`, and :meth:`__str__`
    * In a nutshell:
      * filter_name: A unique identifier, used for lookups and configuration
        files.
      * display_name: Human readable name, used in UI components, will be
        displayed to the user.
      * __str__: Should reuse `display_name` (and may optionally add a summary
        of the filter's configuration parameters), e.g.
        `type(self).display_name() + ' param1=0.1'`
      * apply: must include (at least) the following sanity check:
        ```
        if not self.enabled or image is None:
            return image
        ```
    * A filter must have a default constructor which initializes all internal
      parameters to some sane/recommended defaults.
    * If the filter requires parametrization:
      * Implement setter methods for all parameters which perform proper sanity
        checks.
      * Override :meth:`get_configuration` and :meth:`set_configuration` to
        save & restore the filter's state. Make sure to call the super class'
        get/set configuration methods, too!
    * Each filter class must be registered via :func:`register_filter`
    * Include a test case for your own filters!
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
        """Returns a human-readable name for this filter (to be used in UI components)."""
        return "Basic Filter"

    def __init__(self):
        self.enabled = True

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f'Filter "{type(self).filter_name()}" does not override `apply`.')

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
        return type(self).filter_name()

    def __str__(self) -> str:
        """Returns the string representation.
        
        By convention, this representation must always start with the
        :meth:`display_name`.
        """
        return type(self).display_name()


# List of all available preprocessing operations.
# Since we want to provide a custom ordering of the operations in the UI, this
# list cannot be retrieved automatically (e.g. via inspect)
__REGISTERED_FILTERS = dict()

def register_filter(filter_name: str, cls: FilterBase) -> None:
    """Registers the filter class, so that it can be created via
    :func:`create_filter`.
    
    Args:
        filter_name: Name used to identify this filter.
        cls: Reference to the filter class.

    Raises:
        KeyError: If the `filter_name` has already been used.
    """
    global __REGISTERED_FILTERS
    if filter_name in __REGISTERED_FILTERS:
        raise KeyError(f'Filter name `{filter_name}` has already been registered.')
    if cls == FilterBase or not issubclass(cls, FilterBase):
        raise ValueError(f'Filter {filter_name} is not a FilterBase subclass.')
    __REGISTERED_FILTERS[filter_name] = cls


def unregister_filter(filter_name: str) -> None:
    """Unregisters the filter (used to simplify testing pcc)."""
    global __REGISTERED_FILTERS
    __REGISTERED_FILTERS.pop(filter_name)


def create_filter(filter_name: str) -> FilterBase:
    """Constructs & returns the specified filter.
    
    Raises:
        KeyError: If the `filter_name` is unknown, i.e. if it has not been
            registered via :func:`register_filter`.
    """
    if filter_name not in __REGISTERED_FILTERS:
        raise KeyError(f'Filter with name `{filter_name}` has not been registered!')
    return __REGISTERED_FILTERS[filter_name]()


def list_filter_names() -> list:
    """Returns the names of all registered filters."""
    return [name for name in __REGISTERED_FILTERS]
