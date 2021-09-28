import cv2
import numpy as np
from vito import imutils


class CheckerboardDetector(object):
    def __init__(self, board_specification):
        self.board_spec = board_specification
        self.nrows = self.board_spec.num_inner_corners_vertical
        self.ncols = self.board_spec.num_inner_corners_horizontal

    def process(self, image: np.ndarray):
        image_points = None
        object_points = None
        gray = imutils.grayscale(image)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (self.nrows, self.ncols), flags)
        # If the full checkerboard has been detected, refine corners and store correspondences
        if ret:
            object_points = self.board_spec.reference_points
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            win_size = (5, 5)
            zero_zone = (-1, -1)
            corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
            image_points = corners
            #TODO remove
            vis = image.copy()
            cv2.drawChessboardCorners(vis, (self.nrows, self.ncols), corners, ret)
            dst_sz = (800, 600) if vis.shape[1] > vis.shape[0] else (600, 800)
            vis = cv2.resize(vis, dst_sz)
            from vito import imvis
            imvis.imshow(vis, wait_ms=100)
            #TODO end remove
        return image_points, object_points
    