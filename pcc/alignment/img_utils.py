import cv2
import numpy as np
from vito import imutils


def adjust_size(img, target_width, target_height):
    """
    Center crops and/or pads (border replication) the image to have the
    expected size.
    Currently, only single-channel inputs are supported.
    """
    assert img.ndim == 2
    h, w = img.shape[:2]
    # Adjust height
    delta_h = h - target_height
    if delta_h < 0:
        delta_h *= -1
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top
        if pad_top > 0:
            pad = np.matlib.repmat(img[0, :], pad_top, 1)
            img = np.row_stack((pad, img))
        if pad_bottom > 0:
            pad = np.matlib.repmat(img[-1, :], pad_bottom, 1)
            img = np.row_stack((img, pad))
    elif delta_h > 0:
        offset = delta_h // 2
        img = imutils.crop(img, [0, offset, w, target_height])
    h = img.shape[0]
    # Adjust width
    delta_w = w - target_width
    if delta_w < 0:
        delta_w *= -1
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left
        if pad_left > 0:
            pad = np.matlib.repmat(img[:, 0], 1, pad_left)
            img = np.column_stack((pad, img))
        if pad_right > 0:
            pad = np.matlib.repmat(img[:, -1], 1, pad_right)
            img = np.column_stack((img, pad))
    elif delta_w > 0:
        offset = delta_w // 2
        img = imutils.crop(img, [offset, 0, target_width, h])
    return img


def image_pyramid(src, num_levels):
    """
    Creates the Gaussian image pyramid (each level is upsampled to the original
    src image size).
    """
    target_height, target_width = src.shape[:2]
    # pyrDown requires uint8 inputs
    down_sampled = src.copy()
    pyramid = list()
    # Convert to float (and range [0, 1])
    src = src.astype(float) / 255.0
    pyramid.append(src.copy())
    for lvl in range(num_levels - 1):
        down_sampled = cv2.pyrDown(down_sampled.copy())
        up_sampled = down_sampled.copy()
        for m in range(lvl+1):
            height, width = up_sampled.shape[:2]
            up_sampled = cv2.pyrUp(up_sampled.copy())
        up_sampled = up_sampled.astype(float) / 255.0
        # We must enforce correct size to avoid dimensionality mismatch
        height, width = up_sampled.shape[:2]
        if height != target_height or width != target_width:
            up_sampled = adjust_size(up_sampled, target_width, target_height)
        pyramid.append(up_sampled)
    return pyramid


def image_gradient(image):
    """
    Computes the image gradients (forward differences) dx and dy, returned
    as (W*H x 2) matrix.
    """
    height, width = image.shape[:2]
    dx = np.column_stack((image[:, 1:] - image[:, :-1], np.zeros((height, 1), dtype=float)))
    dy = np.row_stack((image[1:, :] - image[:-1, :], np.zeros((1, width), dtype=float)))
    return np.column_stack((dx.reshape(-1, 1), dy.reshape(-1, 1)))


# def image_gradient_loop(image):
#     height, width = image.shape[:2]
#     dxdy = np.zeros((height*width, 2), dtype=float)
#     for v in range(height):
#         for u in range(width):
#             idx = u + v*width
#             if u+1 == width:
#                 dx = 0
#             else:
#                 dx = image[v, u+1] - image[v, u]
#             if v+1 == height:
#                 dy = 0
#             else:
#                 dy = image[v+1, u] - image[v, u]
#             dxdy[idx, 0] = dx
#             dxdy[idx, 1] = dy
#     return dxdy
