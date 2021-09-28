from . import ClippedCheckerboardSpecification, CheckerboardDetector
import logging
from ..common import PAPER_DIMENSIONS

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    bwidth, bheight = PAPER_DIMENSIONS['A4']
    board = ClippedCheckerboardSpecification('clipped-checkerboard',
                                             board_width_mm=bwidth, board_height_mm=bheight,
                                             num_squares_horizontal=6,
                                             num_squares_vertical=8,
                                             checkerboard_square_length_mm=30)#, overlay_board_specifications=False)
    # from .. import export_board
    # export_board(board, prevent_overwrite=False, export_png=False)
    # # assert False

    
    from vito import imvis

    detector = CheckerboardDetector(board)
    image_points, object_points = detector.process(board.image())
    print('img points', image_points.shape)
    print('ref points', object_points.shape)
    print('img', image_points[:9,0,:])
    print('obj', object_points[:9, :])
    imvis.imshow(board.image(), title='Calibration Board', wait_ms=-1)

    from vito import pyutils
    from ..imgdir import ImageDirectorySource
    import cv2
    src = ImageDirectorySource('example-ccb')
    image_points = list()
    object_points = list()
    img_shape = None
    img_paths_valid = list()
    while src.is_available():
        image, filename = src.next()
        img_shape = image.shape[:2]
        print(f'Processing: {filename}')
        pyutils.tic('Checkerboard Detection')
        pts2d, pts3d = detector.process(image)
        pyutils.toc('Checkerboard Detection')
        if pts2d is None:
            print(f'####### Could not detect the checkerboard!! ######')
        else:
            img_paths_valid.append(filename)
            image_points.append(pts2d)
            object_points.append(pts3d)
    
    rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_shape[::-1], None, None, flags=cv2.CALIB_FIX_K3)
    print('Calibration result:', rms, K, distortion, rvecs, tvecs)
    print(f'fx {K[0,0]}, fy {K[1, 1]}, cx {K[0, 2]}, cy {K[1,2]}')

    h, w = img_shape
    k_undistorted, roi_undistorted = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 1, (w, h))

    error = []
    for i in range(len(object_points)):
        tmp_img_pts, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, distortion)
        err = cv2.norm(image_points[i], tmp_img_pts, cv2.NORM_L2) / len(tmp_img_pts)
        print(f'Reprojection error, img {i}: {err}')
        error.append(err)

    calibration_data = {"rms": rms, "k_distorted": K, "distortion": distortion,
                "k_undistorted": k_undistorted, "roi_undistorted": roi_undistorted,
                "rvecs": rvecs, "tvecs": tvecs,
                "valid_image_paths": img_paths_valid,
                "object_points": object_points,
                "img_points": image_points,
                "calibration_errors": error}