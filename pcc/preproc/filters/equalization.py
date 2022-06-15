import cv2
import numpy as np
from pcc.preproc.filters.filter_base import FilterBase, register_filter


class HistogramEqualization(FilterBase):
    """Applies standard global histogram equalization.

    For RGB images, equalization is applied on the intensity (Y) channel after
    color conversion to YCrCb.
    """

    @staticmethod
    def filter_name() -> str:
        return 'histeq'

    @staticmethod
    def display_name() -> str:
        return 'Histogram Equalization'

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        if image.ndim == 3 and image.shape[2] in [3, 4]:
            ycrcb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2YCrCb)
            yeq = cv2.equalizeHist(ycrcb[:, :, 0])
            ycrcb[:, :, 0] = yeq
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            return cv2.equalizeHist(image)

    def __str__(self) -> str:
        return type(self).display_name()


# class PreProcOpCLAHE(PreProcOperationBase):
#     """Applies contrast limited adaptive histogram equalization."""

#     name = 'clahe'
#     display = 'CLAHE'

#     def __init__(self, clip_limit: float = 2.0, tile_size: typing.Tuple[int, int] = (8, 8)):
#         super().__init__()
#         self.clahe = None
#         self.clip_limit = None
#         self.tile_size = None
#         self.set_clip_limit(clip_limit)
#         self.set_tile_size(tile_size)

#     def description(self) -> str:
#         return f'{self.display} (clip={self.clip_limit:.1f}, tile={self.tile_size})'

#     def apply(self, image: np.ndarray) -> np.ndarray:
#         if not self.enabled:
#             return image
#         if image.ndim == 3 and image.shape[2] in [3, 4]:
#             ycrcb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2YCrCb)
#             yeq = self.clahe.apply(ycrcb[:, :, 0])
#             ycrcb[:, :, 0] = yeq
#             return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
#         else:
#             return self.clahe.apply(image)

#     def _set_clahe(self):
#         self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#     def set_clip_limit(self, clip_limit: float):
#         self.clip_limit = clip_limit
#         self._set_clahe()

#     def set_tile_size(self, tile_size: typing.Tuple[int, int]):
#         self.tile_size = tile_size
#         self._set_clahe()

#     def configure(self, config: dict) -> None:
#         super().configure(config)
#         if 'clip_limit' in config:
#             self.clip_limit = config["clip_limit"]
#         if 'tile_size' in config:
#             self.tile_size = tuple(config["tile_size"])
#         self._set_clahe()

#     def freeze(self) -> dict:
#         d = {'clip_limit': self.clip_limit,
#              'tile_size': self.tile_size}
#         d.update(super().freeze())
#         return d

#     def __repr__(self) -> str:
#         return f'{self.name}(cl={self.clip_limit:.1f}, ts={self.tile_size})'

register_filter(HistogramEqualization.filter_name(), HistogramEqualization)
# register_filter(CLAHE.filter_name(), CLAHE)
