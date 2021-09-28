from PIL.Image import Image
from . import CheckerboardSpecification
from . import CheckerboardDetector

import cv2
from vito import imvis, imutils


import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

class Camera(object):
    def __init__(self, K=[[1, 0, 1], [0, 1, 1], [0, 0, 1]], R=np.eye(3), t=[0, 0, 0]):
        """
        Camera object with intrinsics K and extrinsics R, t
        Extrinsics are in the origin by default
        :param f: focal length
        :param cx: principal point x
        :param cy: principal point y
        :param R: 3x3 ... rotation matrix
        :param t: 3d ... translation vector
        """

        t = np.asarray(t).reshape(3, 1)

        self._K = np.asarray(K)
        self._R_t = np.hstack((R, t))

    def draw_camera(self, ax3d, use_extrinsics=True, scale_camera=1., invert_extrinsics=False):
        """
        Draw camera to a given 3d plot
        :param ax3d: axis of matplotlib 3d plot
        :param use_extrinsics: rotate and translate camera based on extrinsic camera parameters R and t if set True
        """

        tmp_K = self._K * scale_camera
        f, cx, cy = tmp_K[0, 0], tmp_K[0, 2], tmp_K[1, 2]
        cam_origin = np.asarray([0, 0, 0])
        cam_xyz = np.eye(3) * np.min(tmp_K[0:2, 2])  # * self._viz_camera_coordinate_frame_scale
        cam_ray = np.asarray([0, 0, f]) * 2  # * self._viz_focal_length_scale

        pts = np.asarray([np.zeros((3, )) * i/2. for i in range(2)]).squeeze().T
        image_plane_points = np.array([[cx, cx, -cx, -cx], [cy, -cy, -cy, cy], [f, f, f, f]])

        if use_extrinsics:
            tmp_data = np.hstack((cam_origin.reshape(3, 1), cam_xyz, image_plane_points, cam_ray.reshape(3, 1), pts))
            tmp_data_hom = np.vstack((tmp_data, np.ones((1, tmp_data.shape[1]))))

            T = np.vstack((self._R_t, np.eye(4)[3]))
            if invert_extrinsics:
                T = np.linalg.inv(T)
            tmp_data_hom_transformed = T.dot(tmp_data_hom)
            tmp_data = tmp_data_hom_transformed[:3, :] / tmp_data_hom_transformed[3, :]

            cam_origin, cam_xyz, image_plane_points, cam_ray, pts = tmp_data[:, 0].ravel(), tmp_data[:, 1:4], tmp_data[:, 4:8], tmp_data[:, 8].ravel(), tmp_data[:, 9:]

        # plot x, y, z axis in red, green, blue respectively
        all(ax3d.plot(*self._pts2vec(cam_origin, cam_xyz[:, _ax_id]), color=np.eye(3)[_ax_id]) for _ax_id in range(3))  # xyz coordinate frame
        all(ax3d.plot(*self._pts2vec(cam_origin, image_plane_points[:, _ax_id]), color=(0, 0, 0), linewidth=0.5, alpha=0.4) for _ax_id in range(image_plane_points.shape[1]))  # camera origin to image plane edges
        ax3d.plot(*self._pts2vec(cam_origin, cam_ray), color=(0, 0, 0), linewidth=0.7, alpha=0.6)  # cam ray through center of image plane
        all(all(ax3d.plot(*self._pts2vec(image_plane_points[:, from_id], image_plane_points[:, _ax_id]), color=(0, 0, 0), linewidth=0.5, alpha=0.4) for _ax_id in range(from_id, image_plane_points.shape[1])) for from_id in range(image_plane_points.shape[1]))  # connect all image plane points
        ax3d.plot_trisurf(*image_plane_points, alpha=0.2)  # image plane
        ax3d.scatter(pts[0, :], pts[1, :], pts[2, :], color=(0, 0, 0), alpha=0.3)  # rotated points from camera coordinate frame

    def _pts2vec(self, A, B):
        tmp = np.vstack((A, B))
        return [tmp[:, i] for i in range(tmp.shape[1])]

def plot_3d_camera_plots(calib_data):
    def _pts2vec(A, B):
        tmp = np.vstack((A, B))
        return [tmp[:, i] for i in range(tmp.shape[1])]

    origin = np.asarray([0, 0, 0])
    calib_pattern_max = np.max(calib_data['object_points'][0], axis=0)
    origin_xyz = np.eye(3) * np.max(calib_pattern_max)  # scale origin on max length of pattern

    ####################################################################################################################
    # camera centric
    fig = plt.figure("calibration camera centric", figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # scale factor based on x axis length of pattern and principal point along x
    scale_factor = 1 / (calib_data['k_undistorted'][0, 2] / calib_pattern_max[0]) / 5

    camera = Camera(K=calib_data['k_undistorted'])
    camera.draw_camera(ax, use_extrinsics=False, scale_camera=scale_factor)

    (gp_x, gp_y) = calib_pattern_max[:2]
    ground_plane = np.array([[gp_x, gp_x, 0, 0], [gp_y, 0, 0, gp_y], [0, 0, 0, 0]])

    for obj_pts, img_pts, rvec, tvec, img_path in zip(calib_data['object_points'], calib_data['img_points'],
                                                      calib_data['rvecs'], calib_data['tvecs'],
                                                      calib_data['valid_image_paths']):
        T = np.hstack((cv2.Rodrigues(rvec)[0], tvec))
        T = np.vstack((T, np.eye(4)[3]))

        tmp_coordinate_frame = np.hstack((origin.reshape(3, 1), origin_xyz))
        tmp_coordinate_frame = np.vstack((tmp_coordinate_frame, np.ones((1, tmp_coordinate_frame.shape[1]))))
        tmp_coordinate_frame = T.dot(tmp_coordinate_frame)
        tmp_coordinate_frame = tmp_coordinate_frame[:3, :] / tmp_coordinate_frame[3, :]
        tmp_origin, tmp_origin_xyz = tmp_coordinate_frame[:, 0].ravel(), tmp_coordinate_frame[:, 1:]
        all(ax.plot(*_pts2vec(tmp_origin, tmp_origin_xyz[:, _ax_id]), color=np.eye(3)[_ax_id], alpha=0.4) for _ax_id in range(3))

        transformed_ground_plane = np.vstack((ground_plane, np.ones((1, ground_plane.shape[1]))))
        transformed_ground_plane = T.dot(transformed_ground_plane)
        transformed_ground_plane = transformed_ground_plane[:3, :] / transformed_ground_plane[3, :]
        ax.plot_trisurf(*transformed_ground_plane, color=(0, 0, 0), alpha=0.2)  # pattern plane

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt_from_to = np.max(np.abs([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]))
    ax.auto_scale_xyz(*[(-plt_from_to, plt_from_to) for _ in range(3)])

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)

    ####################################################################################################################
    # pattern centric
    fig = plt.figure("calibration pattern centric", figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # plot origin
    all(ax.plot(*_pts2vec(origin, origin_xyz[_ax_id, :]), color=np.eye(3)[_ax_id]) for _ax_id in range(3))

    # scale factor based on x axis length of pattern and principal point along x
    scale_factor = 1 / (calib_data['k_undistorted'][0, 2] / calib_pattern_max[0]) / 10

    print("\n------\ndistances to calibration pattern")
    for obj_pts, img_pts, rvec, tvec, img_path in zip(calib_data['object_points'], calib_data['img_points'], calib_data['rvecs'], calib_data['tvecs'], calib_data['valid_image_paths']):
        tmp_rot = cv2.Rodrigues(rvec)[0]

        t = -tmp_rot.T.dot(tvec)
        R = tmp_rot.T

        print("t:", np.linalg.norm(t))

        camera = Camera(K=calib_data['k_undistorted'], R=R, t=t)
        camera.draw_camera(ax, use_extrinsics=True, scale_camera=scale_factor)  #, invert_extrinsics=True)

    (gp_x, gp_y) = calib_pattern_max[:2]
    ground_plane = np.array([[gp_x, gp_x, 0, 0], [gp_y, 0, 0, gp_y], [0, 0, 0, 0]])
    ax.plot_trisurf(*ground_plane, color=(0, 0, 0), alpha=0.1)  # pattern plane

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt_from_to = np.max(np.abs([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]))
    ax.auto_scale_xyz(*[(-plt_from_to, plt_from_to) for _ in range(3)])

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)

    # plt.show(block=False)
    plt.show()


#TODO remaining calibration pipeline (check shapes) https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html
if __name__ == '__main__':
    board = CheckerboardSpecification('std_checkerboard',
                                      board_width_mm=210, board_height_mm=297,
                                      checkerboard_square_length_mm=25,
                                      num_squares_horizontal=6, num_squares_vertical=10)
    #TODO export demo
    from .. import export_board
    export_board(board, prevent_overwrite=False)
    assert False

    detector = CheckerboardDetector(board)

    from vito import pyutils
    from ..imgdir import ImageDirectorySource
    import os
    src = ImageDirectorySource('cb-example')
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
    plot_3d_camera_plots(calibration_data)


