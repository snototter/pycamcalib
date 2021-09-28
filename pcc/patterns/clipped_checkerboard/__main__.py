from . import *
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    board = ClippedCheckerboardSpecification('checkerboard', num_squares_horizontal=5,
                                  num_squares_vertical=9,
                                  checkerboard_square_length_mm=25)#, overlay_board_specifications=False)
    from .. import export_board
    export_board(board, prevent_overwrite=False)
    assert False

    board = ClippedCheckerboardSpecification('ccb-6x8', num_squares_horizontal=6,
                                  num_squares_vertical=8,
                                  checkerboard_square_length_mm=30)
    from vito import imvis

    imvis.imshow(board.image(), title='Calibration Board', wait_ms=-1)

    detector = ClippedCheckerboardDetector(board)
    image_points, object_points = detector.process(board.image())
    print('img points', image_points.shape)
    print('ref points', object_points.shape)
    print('img', image_points[:4,0,:])
    print('obj', object_points[0, :4, :])

    # from vito import pyutils
    # from ..imgdir import ImageDirectorySource
    # import cv2
    # src = ImageDirectorySource('example-ccb')
    # image_points = list()
    # object_points = list()
    # img_shape = None
    # img_paths_valid = list()
    # while src.is_available():
    #     image, filename = src.next()
    #     img_shape = image.shape[:2]
    #     print(f'Processing: {filename}')
    #     pyutils.tic('Checkerboard Detection')
    #     pts2d, pts3d = detector.process(image)
    #     pyutils.toc('Checkerboard Detection')
    #     if pts2d is None:
    #         print(f'####### Could not detect the checkerboard!! ######')
    #     else:
    #         img_paths_valid.append(filename)
    #         image_points.append(pts2d)
    #         object_points.append(pts3d)
    
    # rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_shape[::-1], None, None, flags=cv2.CALIB_FIX_K3)
    # print('Calibration result:', rms, K, distortion, rvecs, tvecs)
    # print(f'fx {K[0,0]}, fy {K[1, 1]}, cx {K[0, 2]}, cy {K[1,2]}')
