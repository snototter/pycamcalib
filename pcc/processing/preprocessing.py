import cv2
import numpy as np
from vito import imutils


class _PreProcOperation(object):
    def __init__(self):
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = enabled


class PreProcOpGrayscale(_PreProcOperation):
    """Converts an image to grayscale"""
    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.enabled:
            return imutils.grayscale(image)
        return image


#TODO configurable (add config widget)! limit gamma!
class PreProcOpGammaCorrection(_PreProcOperation):
    """Applies Gamma correction"""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = None
        self.lookup_table = None
        self.set_gamma(gamma)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        return cv2.LUT(image, self.lookup_table)

    def set_gamma(self, gamma: float):
        """Precomputes the internal lookup table"""
        self.gamma = gamma
        g_inv = 1.0 / self.gamma
        self.lookup_table = np.clip(np.array([((intensity / 255.0) ** g_inv) * 255
                                             for intensity in np.arange(0, 256)]).astype(np.uint8),
                                    0, 255)


class PreProcOpHistEq(_PreProcOperation):
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



#TODO configurable (tile size, clip limit)
class PreProcOpCLAHE(_PreProcOperation):
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

# examples of different threshold types in cv2: https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
# class PreProcOpThreshold(object):

#TODO maybe add naive contrast adjustment I*alpha + beta https://towardsdatascience.com/exploring-image-processing-techniques-opencv-4860006a243

class Preprocessor(object):
    def __init__(self):
        self.operations = list()
    #TODO list of preprocessor ops

    def add_operation(self, operation: _PreProcOperation):
        self.operations.append(operation)

    def set_enabled(self, index: int, enabled: bool):
        """Enable/disable the operation at the given index."""
        self.operations[index].set_enabled(enabled)

    def process(self, image):
        for op in self.operations:
            image = op.apply(image)
        return image


#TODO remove
if __name__ == '__main__':
    import os
    from vito import imvis
    img = imutils.imread(os.path.join(os.path.dirname(__file__), '..', '..', 'sandbox', 'flamingo.jpg'))

    pp = Preprocessor()
    # pp.add_operation(PreProcOpGrayscale())
    # pp.add_operation(PreProcOpGammaCorrection(2))
    pp.add_operation(PreProcOpHistEq())

    imvis.imshow(img, wait_ms=10)
    img1 = pp.process(img)

    # pp.set_enabled(0, False)
    img2 = pp.process(img)

    # imvis.imshow(img2, 'disabled?', wait_ms=-1)
    imvis.imshow(img1, 'preproc', wait_ms=-1)
