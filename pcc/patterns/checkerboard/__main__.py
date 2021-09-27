from PIL.Image import Image
from . import CheckerboardSpecification
from . import CheckerboardDetector

import cv2
from vito import imvis, imutils

#TODO remaining calibration pipeline (check shapes) https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html
if __name__ == '__main__':
    board = CheckerboardSpecification('cb-10x6', board_width_mm=210, board_height_mm=297,
                                  checkerboard_square_length_mm=25,
                                  num_squares_horizontal=6, num_squares_vertical=10)

    imvis.imshow(board.image(), title='Calibration Board', wait_ms=-1)

    # from .. import export_board
    # export_board(board)

    detector = CheckerboardDetector(board)
    # image = imutils.imread('cb-10x6-example.jpg')
    # image_points, object_points = detector.process(image)
    # print(image_points.shape, object_points.shape)  
    # print(image_points.dtype, object_points.dtype)  
    #TODO assert dtype float32, and shape img pt: Nx1x2, pattern: 1xNx3

    print(board)

    from vito import pyutils
    from ..imgdir import ImageDirectorySource
    import os
    src = ImageDirectorySource('cb-example')
    while src.is_available():
        image = src.next()
        pyutils.tic('Checkerboard Detection')
        image_points, object_points = detector.process(image)
        pyutils.toc('Checkerboard Detection')
