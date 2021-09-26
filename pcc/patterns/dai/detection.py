from .specification import BoardSpecificationDAI
import cv2
import numpy as np
from vito import imutils


class DetectorDAI(object):
    def __init__(self, board_specification: BoardSpecificationDAI):
        self.board_spec = board_specification

    def process(self, image: np.ndarray):
        gray = imutils.grayscale(image)
        ret, corners = cv2.findChessboardCorners(gray, (self.board_spec.num_squares_vertical, self.board_spec.num_squares_horizontal), None)
        cv2.drawChessboardCorners(image, (self.board_spec.num_squares_vertical, self.board_spec.num_squares_horizontal), corners, True)
        from vito import imvis
        imvis.imshow(image, wait_ms=-1)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

if __name__ == '__main__':
    board = BoardSpecificationDAI('dai-5x9', board_width_mm=300, board_height_mm=200,#FIXME 1100//2,
                                  margin_horizontal_mm=100//2, margin_vertical_mm=100//2,
                                  checkerboard_square_length_mm=100//2)
    det = DetectorDAI(board)
    