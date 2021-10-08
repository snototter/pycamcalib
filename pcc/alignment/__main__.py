import cv2
import logging
import numpy as np
from vito import imutils, imvis, pyutils
from .img_alignment import Method, Alignment
import os

_logger = logging.getLogger()

def _generate_warped_image(img, tx, ty, tz, rx, ry, rz):
    H = np.array([[0.93757391, -0.098535322, -8.3316984],
                  [0.0703476, 0.93736351, -32.40559],
                  [-4.9997212e-05, -4.9928687e-05, 1]], dtype=float)
    rows, cols = img.shape[:2]
    warped = cv2.warpPerspective(img, H, (cols, rows), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, H


def demo():
    #TODO separate assets folder, use abspath
    img = imutils.imread(os.path.join(os.path.dirname(__file__), '..', '..', 'sandbox', 'flamingo.jpg'))
    rect = (180, 170, 120, 143)
    target_template = imutils.roi(img, rect)
    imvis.imshow(target_template, 'Template', wait_ms=10)
    warped, H_gt = _generate_warped_image(img, -45, -25, 20, 30, -30, -360)
    imvis.imshow(warped, 'Simulated Warp', wait_ms=10)
    
    # Initial estimate H0
    H0 = np.eye(3, dtype=float)
    H0[0, 2] = rect[0]
    H0[1, 2] = rect[1]
    _logger.info(f'Initial estimate, H0:\n{H0}')

    # print('H0\n', H0)
    # print('H_gt\n', H_gt)

    verbose = True
    pyutils.tic('FC')
    align = Alignment(target_template, Method.FC, full_reference_image=img, num_pyramid_levels=5, verbose=verbose)
    align.set_true_warp(H_gt)
    H_est, result = align.align(warped, H0)
    pyutils.toc('FC')
    imvis.imshow(result, 'Result FC', wait_ms=10)

    pyutils.tic('IC')
    align = Alignment(target_template, Method.IC, full_reference_image=img, num_pyramid_levels=3, verbose=verbose)
    align.set_true_warp(H_gt)
    H_est, result = align.align(warped, H0)
    pyutils.toc('IC')
    imvis.imshow(result, 'Result IC', wait_ms=10)

    pyutils.tic('ESM')
    align = Alignment(target_template, Method.ESM, full_reference_image=img, num_pyramid_levels=5, verbose=verbose)
    align.set_true_warp(H_gt)
    H_est, result = align.align(warped, H0)
    pyutils.toc('ESM')
    imvis.imshow(result, 'Result ESM', wait_ms=-1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
