import cv2
from enum import Enum
import logging
from vito import imutils, imvis, cam_projections as prj
import numpy as np

_logger = logging.getLogger('ImageAlignment')

# Python port of https://github.com/cashiwamochi/LK20_ImageAlignment

class Method(Enum):
    FC = 1  # Forward compositional
    IC = 2  # Inverse compositional
    ESM = 3  # Efficient second-order minimization

    def __str__(self):
        if self == Method.FC:
            return 'Forward Compositional'
        elif self == Method.IC:
            return 'Inverse Compositional'
        elif self == Method.ESM:
            return 'Efficient Second-order Minimization'
        raise NotImplementedError()


def _compute_Jg(sl3_bases):
    """Computes the 9x8 Jacobian Jg."""
    # Paper Eq.(65)
    assert len(sl3_bases) == 8
    Jg = np.zeros((9, 8), dtype=float)
    for i in range(8):
        for j in range(3):
            for k in range(3):
                Jg[j*3+k, i] = sl3_bases[i][j, k]
    return Jg


def _get_SL3_bases():
    """Returns the 8 SL3 bases"""
    bases = list()
    B = np.zeros((3, 3), dtype=float)
    B[0, 2] = 1
    bases.append(B.copy())

    B = np.zeros((3, 3), dtype=float)
    B[1, 2] = 1
    bases.append(B.copy())

    B = np.zeros((3, 3), dtype=float)
    B[0, 1] = 1
    bases.append(B.copy())

    B = np.zeros((3, 3), dtype=float)
    B[1, 0] = 1
    bases.append(B.copy())

    B = np.zeros((3, 3), dtype=float)
    B[0, 0] = 1
    B[1, 1] = -1
    bases.append(B.copy())

    B = np.zeros((3, 3), dtype=float)
    B[1, 1] = -1
    B[2, 2] = 1
    bases.append(B.copy())

    B = np.zeros((3, 3), dtype=float)
    B[2, 0] = 1
    bases.append(B.copy())

    B = np.zeros((3, 3), dtype=float)
    B[2, 1] = 1
    bases.append(B.copy())
    return bases


class Alignment(object):
    def __init__(self, image, H0, method=Method.ESM):
        self.working_image = imutils.grayscale(image)
        self.set_initial_warp(H0)
        self.method = method

        self.blur_kernel_size = (5, 5)
        self.working_image = cv2.GaussianBlur(self.working_image, self.blur_kernel_size, 0)
        #TODO image pyramid!
        
        self.sl3_bases = _get_SL3_bases()
        self.Jg = _compute_Jg(self.sl3_bases)

        _logger.info(f'TODO init method: {method}')
        #TODO precompute!
        imvis.imshow(self.working_image, 'Working Image', wait_ms=-1)
    
    def set_initial_warp(self, H0):
        self.H0 = H0.copy()


def _generate_warped_image(img, tx, ty, tz, rx, ry, rz):
    trans_x = 0.001 * tx
    trans_y = 0.001 * ty
    trans_z = 1 + 0.001 * tz
    R_x = prj.rotx3d(rx*0.1/180.0*np.pi)
    R_y = prj.rotx3d(ry*0.1/180.0*np.pi)
    R_z = prj.rotx3d(rz*0.1/180.0*np.pi)
    # Original code builds the matrix in ZYX (roll-pitch-yaw) order
    R = prj.matmul(R_z, prj.matmul(R_y, R_x))
    t = np.array([trans_x, trans_y, trans_z], dtype=np.float64).reshape((3,1))

    rows, cols = img.shape[:2]
    K = np.array([[1000, 0, cols/2],
                  [0, 1000, rows/2],
                  [0, 0, 1]], dtype=np.float64)

    H = np.zeros_like(R)
    H[0,0] = R[0,0]
    H[1,0] = R[1,0]
    H[2,0] = R[2,0]
    H[0,1] = R[0,1]
    H[1,1] = R[1,1]
    H[2,1] = R[2,1]
    H[0,2] = t[0,0]
    H[1,2] = t[1,0]
    H[2,2] = t[2,0]

    H = prj.matmul(K, prj.matmul(H, np.linalg.inv(K)))
    H /= H[2, 2]
    _logger.info(f'Homography:\n{H}')

    warped = cv2.warpPerspective(img, H, (cols, rows), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, H


def demo():
    img = imutils.imread('lenna.png')
    rect = (210, 210, 160, 160) # TODO check with non-square rect
    target_template = imutils.roi(img, rect)
    vis = img.copy()
    vis = cv2.rectangle(vis, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 3)
    imvis.imshow(vis, 'Input', wait_ms=10)
    imvis.imshow(target_template, 'Template', wait_ms=10)
    warped, H_true = _generate_warped_image(img, -45, -25, 20, 30, -30, -360)
    imvis.imshow(warped, 'Simulated Warp', wait_ms=-1)

    # Initial estimate H0
    H0 = np.eye(3, dtype=float)
    H0[0, 2] = rect[0]
    H0[1, 2] = rect[1]
    _logger.info(f'Initial estimate, H0:\n{H0}')

    img_alignment = Alignment(warped, H0, Method.ESM)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
