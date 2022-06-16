import cv2
import numpy as np
from vito import imutils
from pcc.preproc.filters.filter_base import FilterBase, register_filter


class Thresholding(FilterBase):
    """Applies either fixed thresholding or chooses the optimal threshold via
    Otsu's or the Triangle algorithm."""

    @staticmethod
    def filter_name() -> str:
        return 'threshold'

    @staticmethod
    def display_name() -> str:
        return 'Thresholding'
#TODO doc should ref to permalink for: https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
    THRESHOLD_TYPES = {
        'binary': cv2.THRESH_BINARY,
        'binary-inv': cv2.THRESH_BINARY_INV,
        'to-zero': cv2.THRESH_TOZERO,
        'to-zero-inv': cv2.THRESH_TOZERO_INV,
        'truncate': cv2.THRESH_TRUNC
    }

    METHODS = {
        'global': 0,
        'otsu': cv2.THRESH_OTSU,
        'triangle': cv2.THRESH_TRIANGLE
    }

    def __init__(self):
        super().__init__()
        self.threshold_value = None
        self.threshold_type = None
        self.method = None
        self.max_value = None
        self.set_threshold_value(127)
        self.set_threshold_type('binary')
        self.set_method('global')
        self.set_max_value(255)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled or image is None:
            return image
        # Otsu & Triangle algorithms only support grayscale images
        if image.ndim == 3 and self.method in [cv2.THRESH_TRIANGLE, cv2.THRESH_OTSU]:
            image = imutils.grayscale(image)
        _, thresh = cv2.threshold(image, self.threshold_value, self.max_value,
            self.threshold_type + self.method)
        return thresh

    def set_threshold_value(self, threshold: int) -> None:
        self.threshold_value = threshold

    def set_max_value(self, max_value: int) -> None:
        self.max_value = max_value

    def set_threshold_type(self, ttype: str) -> None:
        if ttype not in Thresholding.THRESHOLD_TYPES:
            raise ValueError(f'Threshold type "{ttype}" is not supported!')
        self.threshold_type = Thresholding.THRESHOLD_TYPES[ttype]

    def set_method(self, method: str) -> None:
        if method not in Thresholding.METHODS:
            raise ValueError(f'Threshold method "{method}" is not supported!')
        self.method = Thresholding.METHODS[method]

    def set_configuration(self, config: dict) -> None:
        super().set_configuration(config)
        if 'threshold_value' in config:
            self.threshold_value = config["threshold_value"]
        if 'type' in config:
            self.set_threshold_type(config["type"])
        if 'max_value' in config:
            self.max_value = config["max_value"]
        if 'method' in config:
            self.set_method(config["method"])

    def get_configuration(self) -> dict:
        d = {
            'type': self._type2str(),
            'max_value': self.max_value,
            'method': self._method2str()
        }
        # Otsu & Triangle compute the optimal threshold based
        # on the image content (a fixed threshold will simply
        # be ignored)
        if self.method not in [cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE]:
            d['threshold_value'] = self.threshold_value
        d.update(super().get_configuration())
        return d

    def _type2str(self, threshold_type: int = None) -> str:
        if threshold_type is None:
            threshold_type = self.threshold_type
        for ts, tt in Thresholding.THRESHOLD_TYPES.items():
            if tt == threshold_type:
                return ts
        raise ValueError(f'Threshold type ({threshold_type}) is not supported!')

    def _method2str(self, method: int = None) -> str:
        if method is None:
            method = self.method
        for ms, mt in Thresholding.METHODS.items():
            if mt == method:
                return ms
        raise ValueError(f'Threshold method ({method}) is not supported!')

    def __str__(self) -> str:
        return f'{type(self).display_name()} (t={self.threshold_value}, max={self.max_value}, {self._type2str()}, {self._method2str()})'



class AdaptiveThresholding(FilterBase):
    """Applies adaptive thresholding"""

    @staticmethod
    def filter_name() -> str:
        return 'adaptive-threshold'

    @staticmethod
    def display_name() -> str:
        return 'Adaptive Thresholding'

    THRESHOLD_TYPES = {
        'binary': cv2.THRESH_BINARY,
        'binary-inv': cv2.THRESH_BINARY_INV
    }

    METHODS = {
        'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
        'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }

    def __init__(self):
        super().__init__()
        self.max_value = None
        self.method = None
        self.threshold_type = None
        self.block_size = None
        self.C = None
        self.set_max_value(255)
        self.set_method('mean')
        self.set_threshold_type('binary')
        self.set_block_size(5)
        self.set_C(0)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled or image is None:
            return image
        # Ensure single-channel input
        if image.ndim == 3:
            image = imutils.grayscale(image)
        return cv2.adaptiveThreshold(image, self.max_value, self.method, self.threshold_type, self.block_size, self.C)

    def set_max_value(self, max_value: int) -> None:
        self.max_value = max_value

    def set_threshold_type(self, ttype: str) -> None:
        if ttype not in AdaptiveThresholding.THRESHOLD_TYPES:
            raise ValueError(f'Threshold type "{ttype}" is not supported!')
        self.threshold_type = AdaptiveThresholding.THRESHOLD_TYPES[ttype]

    def set_method(self, method: str) -> None:
        if method not in AdaptiveThresholding.METHODS:
            raise ValueError(f'Threshold method "{method}" is not supported!')
        self.method = AdaptiveThresholding.METHODS[method]

    def set_C(self, C: int) -> None:
        self.C = C

    def set_block_size(self, block_size: int) -> None:
        if block_size % 2 == 0:
            raise ValueError(f'Block size ({block_size}) must be odd!')
        self.block_size = block_size

    def set_configuration(self, config: dict) -> None:
        super().set_configuration(config)
        if 'method' in config:
            self.set_method(config['method'])
        if 'type' in config:
            self.set_threshold_type(config['type'])
        if 'max_value' in config:
            self.set_max_value(config['max_value'])
        if 'block_size' in config:
            self.set_block_size(config['block_size'])
        if 'C' in config:
            self.set_C(config['C'])

    def get_configuration(self) -> dict:
        d = {
            'max_value': self.max_value,
            'method': self._method2str(),
            'threshold_type': self._type2str(),
            'block_size': self.block_size,
            'C': self.C
        }
        d.update(super().get_configuration())
        return d

    def _type2str(self, threshold_type: int = None) -> str:
        if threshold_type is None:
            threshold_type = self.threshold_type
        for ts, tt in AdaptiveThresholding.THRESHOLD_TYPES.items():
            if tt == threshold_type:
                return ts
        raise ValueError(f'Adaptive threshold type ({threshold_type}) is not supported!')

    def _method2str(self, method: int = None) -> str:
        if method is None:
            method = self.method
        for ms, mt in AdaptiveThresholding.METHODS.items():
            if mt == method:
                return ms
        raise ValueError(f'Adaptive threshold method ({method}) is not supported!')

    def __str__(self) -> str:
        return f'{type(self).display_name()} (max={self.max_value}, {self._method2str()}, {self._type2str()}, {self.block_size}x{self.block_size}, C={self.C:.1f})'


register_filter(Thresholding.filter_name(), Thresholding)
register_filter(AdaptiveThresholding.filter_name(), AdaptiveThresholding)
