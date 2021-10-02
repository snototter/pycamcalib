import cv2
import logging
import numpy as np
import toml
import typing
from vito import imutils

#TODO implement additional ops:
# * threshold variants, see e.g. https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
# * maybe add naive contrast adjustment I*alpha + beta https://towardsdatascience.com/exploring-image-processing-techniques-opencv-4860006a243


_logger = logging.getLogger('Preprocessing')


class ConfigurationError(Exception):
    """Error upon loading or adjusting parameters of a preprocessor operations."""
    pass


class PreProcOperationBase(object):
    """Base class of preprocessing operations.
    
Basic requirements for each derived class:
* it must provide a unique class-wide 'name' attribute
* it must provide a class-wide 'display' attribute (speaking name for class selection before instantiation)
* it must implement 'description() -> str' to return a brief but nicely formatted name (for UI display)
* it must implement 'apply(np.ndarray) -> np.ndarray'
* it must be registered within 'AVAILABLE_PREPROCESSOR_OPERATIONS'

If a derived class requires additional parameters:
* it's constructor must initialize all parameters to sane default values
* it must override 'configure(dict) -> None' to load parameters from a dictionary
* it must override 'freeze() -> dict' to store all parameters in a dictionary
* it should override '__repr__() -> str'
    """
    def __init__(self):
        self.enabled = True

    def set_enabled(self, enabled) -> None:
        self.enabled = enabled

    def __repr__(self) -> str:
        return self.name

    def configure(self, config: dict) -> None:
        # The 'enabled' field is optional in TOML configurations.
        if 'enabled' in config:
            self.enabled = config['enabled']

    def freeze(self) -> dict:
        d = {'name': self.name}
        # Operations are enabled by default. Thus, we omit this field
        # for TOML serialization
        if not self.enabled:
            d['enabled'] = self.enabled
        return d


class PreProcOpGrayscale(PreProcOperationBase):
    """Converts an image to grayscale"""

    name = 'grayscale'
    display = 'Grayscale'
    
    def description(self) -> str:
        return self.display

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.enabled:
            return imutils.grayscale(image)
        return image


#TODO configurable (add config widget)! limit gamma!
class PreProcOpGammaCorrection(PreProcOperationBase):
    """Applies Gamma correction"""

    name = 'gamma'
    display = 'Gamma Correction'

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = None
        self.lookup_table = None
        self.set_gamma(gamma)

    def description(self) -> str:
        return f'{self.display} (g={self.gamma:.1f})'

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        return cv2.LUT(image, self.lookup_table)

    def set_gamma(self, gamma: float) -> None:
        """Precomputes the internal lookup table"""
        self.gamma = gamma
        g_inv = 1.0 / self.gamma
        self.lookup_table = np.clip(np.array([((intensity / 255.0) ** g_inv) * 255
                                             for intensity in np.arange(0, 256)]).astype(np.uint8),
                                    0, 255)

    def configure(self, config: dict) -> None:
        super().configure(config)
        self.set_gamma(config['gamma'])

    def freeze(self) -> dict:
        d = {'gamma': self.gamma}
        d.update(super().freeze())
        return d

    def __repr__(self) -> str:
        return f'{self.name}(g={self.gamma:.1f})'


class PreProcOpHistEq(PreProcOperationBase):
    """Applies standard (fixed) histogram equalization.

For RGB images, equalization is applied on the intensity (Y) channel after
color conversion to YCrCb."""

    name = 'histeq'
    display = 'Histogram Equalization'

    def description(self) -> str:
        return self.display

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            yeq = cv2.equalizeHist(ycrcb[:, :, 0])
            ycrcb[:, :, 0] = yeq
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            return cv2.equalizeHist(image)


#TODO configurable (tile size, clip limit), add widget!
class PreProcOpCLAHE(PreProcOperationBase):
    """Applies contrast limited adaptive histogram equalization."""

    name = 'clahe'
    display = 'CLAHE'

    def __init__(self, clip_limit: float = 2.0, tile_size: typing.Tuple[int, int] = (8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.clahe = None
        self._set_clahe()

    def description(self) -> str:
        return f'{self.display} (clip={self.clip_limit:.1f}, tile={self.tile_size})'

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            yeq = self.clahe.apply(ycrcb[:, :, 0])
            ycrcb[:, :, 0] = yeq
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            return self.clahe.apply(image)

    def _set_clahe(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def set_clip_limit(self, clip_limit: float):
        self.clip_limit = clip_limit
        self._set_clahe()

    def set_tile_size(self, tile_size: typing.Tuple[int, int]):
        self.tile_size = tile_size
        self._set_clahe()

    def configure(self, config: dict) -> None:
        super().configure(config)
        if 'clip_limit' in config:
            self.clip_limit = config["clip_limit"]
        if 'tile_size' in config:
            self.tile_size = tuple(config["tile_size"])
        self._set_clahe()

    def freeze(self) -> dict:
        d = {'clip_limit': self.clip_limit,
             'tile_size': self.tile_size}
        d.update(super().freeze())
        return d

    def __repr__(self) -> str:
        return f'{self.name}(cl={self.clip_limit:.1f}, ts={self.tile_size})'


# List of all available preprocessing operations.
# Since we want to provide a custom ordering of the operations in the UI, this
# list cannot be retrieved automatically (e.g. via inspect)
AVAILABLE_PREPROCESSOR_OPERATIONS = [
    PreProcOpGrayscale, PreProcOpGammaCorrection,
    PreProcOpHistEq, PreProcOpCLAHE
]

## Removed because we want a custom ordering within the UI combobox
# def generate_operation_mapping():
#     """Returns a 'name':class mapping for all preprocessing operations
#     defined within this module."""
#     operations = dict()
#     for name, obj in inspect.getmembers(sys.modules[__name__]):
#         if inspect.isclass(obj) and name.startswith('PreProcOp'):
#             operations[obj.name] = obj
#     return operations


class Preprocessor(object):
    """Implements the preprocessing pipeline.

This is the main "workflow" class, i.e. it holds a list of operations which are
applied subsequently to a given image via 'process()'.
    """
    def __init__(self):
        self.operations = list()
        self._operation_map = {opcls.name: opcls for opcls in AVAILABLE_PREPROCESSOR_OPERATIONS}

    def add_operation(self, operation: PreProcOperationBase):
        _logger.info(f'Adding operation #{len(self.operations)}: {operation}')
        self.operations.append(operation)

    def set_enabled(self, index: int, enabled: bool):
        """Enable/disable the operation at the given index."""
        self.operations[index].set_enabled(enabled)

    def swap_previous(self, index: int):
        """Swap operation at index with operation at index-1."""
        self.operations[index], self.operations[index-1] = self.operations[index-1], self.operations[index]
    
    def swap_next(self, index: int):
        """Swap operation at index with operation at index+1."""
        self.operations[index], self.operations[index+1] = self.operations[index+1], self.operations[index]

    def remove(self, index: int):
        """Remove operation at index"""
        del self.operations[index]

    def process(self, image):
        i = 0
        for op in self.operations:
            image = op.apply(image)
            imvis.imshow(image, f'prepoc step #{i}', wait_ms=10)
            i+=1
        return image

    def freeze(self):
        """Returns a dictionary which can be used to restore the current state
        of this preprocessing pipeline.
        Nested dict structure is intended to allow a) having a separate configuration
        file for this pipeline and b) merge it with other (sub)module configurations.
        """
        return {'preprocessing': {'operations': [op.freeze() for op in self.operations]}}

    def saveTOML(self, filename):
        """Stores the current state of this preprocessing pipeline to disk."""
        with open(filename, 'w') as fp:
            toml.dump(self.freeze(), fp)

    def loadTOML(self, filename):
        """Loads the preprocessing pipeline from the given TOML file.
        The operations must be configured as [[preprocessing.operations]]
        entries."""
        _logger.info(f'Trying to load preprocessing pipeline from {filename}')
        try:
            config = toml.load(filename)
            self.operations.clear()
            for op_config in config['preprocessing']['operations']:
                operation = self._operation_map[op_config['name']]()
                if len(op_config) > 1:
                    operation.configure(op_config)
                self.add_operation(operation)
        except KeyError as e:
            raise ConfigurationError(f"Invalid/missing configuration key '{str(e)}'") from None
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: '{filename}'") from None
        except (toml.TomlDecodeError, TypeError) as e:
            raise ConfigurationError(f"Invalid TOML configuration: {e}") from None


#TODO clean up or remove
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # op_map = generate_operation_mapping()

    import os
    from vito import imvis
    img = imutils.imread(os.path.join(os.path.dirname(__file__), '..', '..', 'sandbox', 'flamingo.jpg'))

    pp = Preprocessor()
    pp.loadTOML(os.path.join(os.path.dirname(__file__), '..', '..', 'sandbox', 'sandbox.toml'))
    # pp.add_operation(op_map['histeq']())
    # pp.add_operation(PreProcOpGammaCorrection(2))
    # pp.add_operation(PreProcOpHistEq())
    # pp.add_operation(PreProcOpGrayscale())

    imvis.imshow(img, wait_ms=10)
    img1 = pp.process(img)

    # pp.set_enabled(0, False)
    img2 = pp.process(img)

    # imvis.imshow(img2, 'disabled?', wait_ms=-1)
    imvis.imshow(img1, 'preproc', wait_ms=-1)
