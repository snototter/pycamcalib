import numpy as np
from vito import imutils
from pcc.preproc.filters.filter_base import FilterBase, register_filter

class GrayscaleConversion(FilterBase):
    """Converts an image to grayscale"""

    @staticmethod
    def filter_name() -> str:
        return 'grayscale'

    @staticmethod
    def display_name() -> str:
        return 'Grayscale'

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.enabled:
            return imutils.grayscale(image)
        return image


class GammaCorrection(FilterBase):
    """Applies Gamma correction"""

    @staticmethod
    def filter_name() -> str:
        return 'gamma'

    @staticmethod
    def display_name() -> str:
        return 'Gamma Correction'

    def __init__(self):
        super().__init__()
        self.gamma = None
        self.lookup_table = None
        self.set_gamma(1.0)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        return cv2.LUT(image, self.lookup_table)

    def set_gamma(self, gamma: float) -> None:
        """Precomputes the internal lookup table"""
        self.gamma = float(gamma)
        g_inv = 1.0 / self.gamma
        self.lookup_table = np.clip(
            np.array([((intensity / 255.0) ** g_inv) * 255
                for intensity in np.arange(0, 256)]).astype(np.uint8),
            0, 255)

    def set_configuration(self, config: dict) -> None:
        super().set_configuration(config)
        self.set_gamma(config['gamma'])

    def get_configuration(self) -> dict:
        d = {'gamma': self.gamma}
        d.update(super().get_configuration())
        return d

    def __str__(self) -> str:
        return f'{type(self).display_name()} (g={self.gamma:.1f})'



register_filter(GrayscaleConversion.filter_name(), GrayscaleConversion)
register_filter(GammaCorrection.filter_name(), GammaCorrection)

