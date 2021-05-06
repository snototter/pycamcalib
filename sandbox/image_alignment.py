import cv2
from enum import Enum
import logging
from vito import imutils, imvis, cam_projections as prj
from vito import pyutils as pu
import numpy as np
import numpy.matlib

_logger = logging.getLogger('ImageAlignment')

# Python port (and speed improvements) of https://github.com/cashiwamochi/LK20_ImageAlignment

#TODO list:
# * test with eddy
# * refactor eddy

#TODO replace matmul by prj.matmul
def matmul(A, B):
    if A.ndim == 1:
        raise RuntimeError('1Dim inputs FIRST!!!')
    if B.ndim == 1:
        raise RuntimeError('1Dim inputs SECOND!!!')
    return np.matmul(A, B)


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
    for col in range(8):
        for j in range(3):
            for k in range(3):
                Jg[j*3+k, col] = sl3_bases[col][j, k]
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


def _adjust_size(img, target_width, target_height):
    """Center crops and/or pads (with border row/column) the image to have the expected size."""
    h, w = img.shape[:2]
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


def _image_pyramid(src, num_levels):
    """Creates the Gaussian image pyramid (each level is upsampled to the original src image size)."""
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
            up_sampled = _adjust_size(up_sampled, target_width, target_height)
        pyramid.append(up_sampled)
    return pyramid


def _image_gradient(image):
    height, width = image.shape[:2]
    dx = np.column_stack((image[:, 1:] - image[:, :-1], np.zeros((height, 1), dtype=float)))
    dy = np.row_stack((image[1:, :] - image[:-1, :], np.zeros((1, width), dtype=float)))
    return np.column_stack((dx.reshape(-1, 1), dy.reshape(-1, 1)))


def _image_gradient_loop(image):
    height, width = image.shape[:2]
    dxdy = np.zeros((height*width, 2), dtype=float)
    for v in range(height):
        for u in range(width):
            idx = u + v*width
            if u+1 == width:
                dx = 0
            else:
                dx = image[v, u+1] - image[v, u]
            if v+1 == height:
                dy = 0
            else:
                dy = image[v+1, u] - image[v, u]
            dxdy[idx, 0] = dx
            dxdy[idx, 1] = dy
    return dxdy


class Alignment(object):
    def __init__(self, image, method=Method.ESM, num_pyramid_levels=4,
                 blur_kernel_size=(5, 5), max_iterations=50,
                 verbose=False, full_reference_image=None):
        # TODO if verbose is True, you must provide the full reference image, too!
        self.verbose = verbose
        self.full_reference_image = full_reference_image
        self.template_image = imutils.grayscale(image)
        self.height, self.width = image.shape[:2]
        self.H0 = None
        self.H_gt = None
        self.method = method
        self.max_iterations = max_iterations
        self.num_pyramid_levels = num_pyramid_levels
        self.blur_kernel_size = blur_kernel_size
        self.template_image = cv2.GaussianBlur(self.template_image, self.blur_kernel_size, 0)
        self.template_pyramid = _image_pyramid(self.template_image, self.num_pyramid_levels)      
        self.sl3_bases = _get_SL3_bases()
        self.Jg = _compute_Jg(self.sl3_bases)
        self.JwJg = None
        self.dxdys = None  # Stores precomputed gradients for each pyramid level
        self.Js = None
        self.Hs = None
        self._precompute()

    def set_true_warp(self, H_gt):
        self.H_gt = H_gt.copy()
        if self.full_reference_image is None:
            _logger.error('To show the progress, you must provide the full reference image, too!')

    def align(self, image, H0):
        self.H0 = H0
        if self.verbose:
            vis = self._warp_current_image(image, np.eye(3))
            imvis.imshow(vis, 'Initial Warp', wait_ms=10)


        curr_original_image = image.copy()
        working_image = imutils.grayscale(image)
        working_image = cv2.GaussianBlur(working_image, self.blur_kernel_size, 0)
        working_pyramid = _image_pyramid(working_image, self.num_pyramid_levels)

        H = np.eye(3, dtype=float)
        # Coarse-to-fine:
        for lvl in range(self.num_pyramid_levels):
            pyr_lvl = self.num_pyramid_levels - lvl - 1
            H = self._process_in_layer(H, working_pyramid[pyr_lvl],
                                       self.template_pyramid[pyr_lvl], pyr_lvl)

        warped = self._warp_current_image(curr_original_image, H)
        return H, warped

    def _compute_Hessian(self, J):
        # pu.tic('hess-vec')
        Hess = matmul(np.transpose(J), J)
        # pu.toc('hess-vec')
        # Sped up from ~100ms (loop version) to 0.4ms
        # pu.tic('hess-loop')
        # num_params = len(self.sl3_bases)
        # Hessian = np.zeros((num_params, num_params), dtype=float)
        # for r in range(J.shape[0]):
        #     row = J[r, :].reshape((1, -1))
        #     Hessian += matmul(np.transpose(row), row)
        # pu.toc('hess-loop')
        # for r in range(8):
        #     for c in range(8):
        #         if Hess[r,c] != Hessian[r,c]:
        #             print(f'HESSIAN DIFFERS AT {r},{c}: {Hess[r,c]-Hessian[r,c]} {Hess[r,c]} vs {Hessian[r,c]}')
        return Hess

    def _compute_Jacobian(self, dxdy, ref_dxdy=None):
        # Sped up from 70ms (loop version) to 0.8/0.9ms
        dim_g3d = (self.width*self.height, 1, 2)
        if self.method == Method.ESM:
            assert ref_dxdy is not None
            grad = ((dxdy + ref_dxdy) / 2.0).reshape(dim_g3d)
        elif self.method in [Method.FC, Method.IC]:
            grad = dxdy.reshape(dim_g3d)
        else:
            raise NotImplementedError()
        J3d = matmul(grad, self.JwJg)
        J = J3d.reshape((self.height * self.width, self.JwJg.shape[2]))
        return J
        # J = np.zeros((self.height*self.width, 8), dtype=float)
        # if self.method == Method.ESM:
        #     assert ref_dxdy is not None
        #     Ji = (dxdy + ref_dxdy) / 2.0

        # pu.tic('double loop')
        # for u in range(self.height):
        #     for v in range(self.width):
        #         idx = u*self.width + v
        #         if self.method in [Method.FC, Method.IC]:
        #             dd_row = dxdy[idx, :].reshape((1, -1))
        #             J[idx, :] = matmul(dd_row, self.JwJg_list[idx])
        #         elif self.method == Method.ESM:
        #             Ji_row = Ji[idx, :].reshape((1, -1))
        #             J[idx, :] = matmul(Ji_row, self.JwJg_list[idx])
        #         else:
        #             raise NotImplementedError()
        # pu.toc('double loop')
        # pu.tic('np')
        # if self.method in [Method.FC, Method.IC]:
        #     grad = dxdy.reshape((self.width*self.height, 1, 2))
        #     print(f'MULTIPLYING {grad.shape} * {self.JwJg.shape}')
        #     j = matmul(grad, self.JwJg)
        # else:
        #     raise NotImplementedError()
        # pu.toc('np')
        # x = j.reshape((self.width*self.height, 8))
        # print(j.shape)
        # for i in range(self.height*self.width):
        #     if not np.array_equal(x[i,:], J[i,:]):
        #         print(f'Mismatch at pixel idx {i}')
        # assert np.array_equal(x, J)
        # return J

    def _compute_JwJg(self):
        # Sped up from 170ms to 6ms
        # pu.tic('loop')
        # self.JwJg_list = list()
        # for v in range(self.height):
        #     for u in range(self.width):
        #         # Eq.(63)
        #         Jw = np.array([
        #                        [u, v, 1, 0, 0, 0, -u*u, -u*v, -u],
        #                        [0, 0, 0, u, v, 1, -u*v, -v*v, -v]],
        #                       dtype=float)
        #         # Shapes: [2x8] = [2x9] * [9x8]
        #         JwJg = matmul(Jw, self.Jg)
        #         self.JwJg_list.append(JwJg)
        # pu.toc('loop')
        # pu.tic('3d')
        # jwjg = np.zeros(self.height*self.width, 2, 8)
        u, v = np.meshgrid(np.arange(0, self.width), np.arange(0, self.height))
        u = u.reshape((-1, ))
        v = v.reshape((-1, ))
        jw = np.zeros((self.height*self.width, 2, 9), dtype=float)
        jw[:, 0, 0] = u
        jw[:, 0, 1] = v
        jw[:, 0, 2] = 1
        jw[:, 0, 6] = -np.multiply(u, u)
        jw[:, 0, 7] = -np.multiply(u, v)
        jw[:, 0, 8] = -u
        jw[:, 1, 3] = u
        jw[:, 1, 4] = v
        jw[:, 1, 5] = 1
        jw[:, 1, 6] = -np.multiply(u, v)
        jw[:, 1, 7] = -np.multiply(v, v)
        jw[:, 1, 8] = -v

        jgshape = self.Jg.shape
        jg = self.Jg.reshape((1, *jgshape))
        self.JwJg = matmul(jw, jg)

    def _compute_residuals(self, cur_image, ref_image):
        res = 0.0
        residuals = np.zeros((self.height*self.width, 1), dtype=float)
        for v in range(self.height):
            for u in range(self.width):
                idx = v*self.width + u
                r = 0
                if self.method in [Method.IC, Method.ESM]:
                    r = cur_image[v, u] - ref_image[v, u]
                elif self.method == Method.FC:
                    r = ref_image[v, u] - cur_image[v, u]
                else:
                    raise NotImplementedError()
                residuals[idx, 0] = r
                res += r*r
        return residuals, np.sqrt(res / (self.height * self.width))

    def _compute_update_params(self, hessian, J, residuals):
        params = np.zeros((8, 1), dtype=float)
        hessian_inv = np.linalg.inv(hessian)

        for v in range(self.height):
            for u in range(self.width):
                idx = u + v*self.width
                J_row = J[idx, :].reshape((1, -1))
                if self.method == Method.ESM:
                    params += -1 * np.transpose(J_row) * residuals[idx, 0]
                else:
                    params += np.transpose(J_row) * residuals[idx, 0]
        params = matmul(hessian_inv, params)
        return params

    def _is_converged(self, curr_error, prev_error):
        if prev_error < 0:
            return False
        if prev_error < curr_error + 1e-6: #0.0000001: # TODO check numerical stability
            return True
        return False

    def _precompute(self):
        if self.method == Method.FC:
            self._compute_JwJg()
        elif self.method == Method.IC:
            self._compute_JwJg()
            self.dxdys = list()
            self.Js = list()
            self.Hs = list()
            for lvl in range(self.num_pyramid_levels):
                dxdy = _image_gradient(self.template_pyramid[lvl])
                self.dxdys.append(dxdy.copy())
                J = self._compute_Jacobian(dxdy)
                self.Js.append(J.copy())
                self.Hs.append(self._compute_Hessian(J).copy())
        elif self.method == Method.ESM:
            self._compute_JwJg()
            self.dxdys = [_image_gradient(self.template_pyramid[lvl]) for lvl in range(self.num_pyramid_levels)]
        else:
            raise NotImplementedError()

    def _helper_process_fc(self, tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level):
        prev_error = -1
        curr_error = -1
        H = tmp_H.copy()
        for iteration in range(self.max_iterations):
            curr_working_image = self._warp_current_image(curr_image_pyramid, H)
            # imvis.imshow(cur_working_image, "warped current image", wait_ms=-1)
            residuals, curr_error = self._compute_residuals(curr_working_image, ref_image_pyramid)
            _logger.info(f'Iteration[{iteration:3d}] Level[{pyramid_level}]: {curr_error:.6f}')
            dxdy = _image_gradient(curr_working_image)
            J = self._compute_Jacobian(dxdy, None)
            Hessian = self._compute_Hessian(J)
            update_params = self._compute_update_params(Hessian, J, residuals)
            H_new = self._update_warp(update_params, H)

            if prev_error < 0:
                # Decide whether iterative update should be done in this pyramid level or not
                curr_working_image = self._warp_current_image(curr_image_pyramid, H_new)
                prev_error = curr_error
                residuals, curr_error = self._compute_residuals(curr_working_image, ref_image_pyramid)
                if self._is_converged(curr_error, prev_error):
                    break
                else:
                    H = H_new.copy()
                    if self.verbose:
                        self._show_progress(H)
                    continue  # that's actually important!
            
            if self._is_converged(curr_error, prev_error):
                break
            else:
                prev_error = curr_error
                H = H_new.copy()
                if self.verbose:
                    self._show_progress(H)
        return H, curr_error
    
    def _helper_process_ic(self, tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level):
        prev_error = -1
        curr_error = -1
        H = tmp_H.copy()
        for iteration in range(self.max_iterations):
            curr_working_image = self._warp_current_image(curr_image_pyramid, H)
            residuals, curr_error = self._compute_residuals(curr_working_image, ref_image_pyramid)
            _logger.info(f'Iteration[{iteration:3d}] Level[{pyramid_level}]: {curr_error:.6f}')
            update_params = self._compute_update_params(self.Hs[pyramid_level], self.Js[pyramid_level], residuals)
            H_new = self._update_warp(update_params, H)

            if prev_error < 0:
                # Decide whether iterative update should be done in this pyramid level or not
                curr_working_image = self._warp_current_image(curr_image_pyramid, H_new)
                prev_error = curr_error
                residuals, curr_error = self._compute_residuals(curr_working_image, ref_image_pyramid)
                if self._is_converged(curr_error, prev_error):
                    break
                else:
                    H = H_new.copy()
                    if self.verbose:
                        self._show_progress(H)
                    continue  # that's actually important!
                    
            if self._is_converged(curr_error, prev_error):
                break
            else:
                prev_error = curr_error
                H = H_new.copy()
                if self.verbose:
                    self._show_progress(H)
        return H, curr_error
    
    def _helper_process_esm(self, tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level):
        prev_error = -1
        curr_error = -1
        H = tmp_H.copy()
        for iteration in range(self.max_iterations):
            curr_working_image = self._warp_current_image(curr_image_pyramid, H)
            residuals, curr_error = self._compute_residuals(curr_working_image, ref_image_pyramid)
            _logger.info(f'Iteration[{iteration:3d}] Level[{pyramid_level}]: {curr_error:.6f}')
            dxdy = _image_gradient(curr_working_image)
            J = self._compute_Jacobian(dxdy, self.dxdys[pyramid_level])
            Hess = self._compute_Hessian(J)
            update_params = self._compute_update_params(Hess, J, residuals)
            H_new = self._update_warp(update_params, H)

            if prev_error < 0:
                # Decide whether iterative update should be done in this pyramid level or not
                curr_working_image = self._warp_current_image(curr_image_pyramid, H_new)
                prev_error = curr_error
                residuals, curr_error = self._compute_residuals(curr_working_image, ref_image_pyramid)
                if self._is_converged(curr_error, prev_error):
                    break
                else:
                    H = H_new.copy()
                    if self.verbose:
                        self._show_progress(H)
                    continue  # that's actually important!
            
            if self._is_converged(curr_error, prev_error):
                break
            else:
                prev_error = curr_error
                H = H_new.copy()
                if self.verbose:
                    self._show_progress(H)
        return H, curr_error

    def _process_in_layer(self, tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level):
        # pu.tic('proc-layer')
        if self.method == Method.FC:
            H, error = self._helper_process_fc(tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level)
        elif self.method == Method.IC:
            H, error = self._helper_process_ic(tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level)
        elif self.method == Method.ESM:
            H, error = self._helper_process_esm(tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level)
        else:
            raise NotImplementedError()
        # pu.toc('proc-layer')
        _logger.info(f'{self.method}, final residual on pyramid level [{pyramid_level}]: {error}')
        return H

    def _show_progress(self, H):
        if self.full_reference_image is None:
            _logger.error('Full template image was not set, cannot show progress!')
            return
        pts = np.zeros((3, 4), dtype=float)
        pts[:, 0] = np.array([0, 0, 1], dtype=float)
        pts[:, 1] = np.array([self.width, 0, 1], dtype=float)
        pts[:, 2] = np.array([self.width, self.height, 1], dtype=float)
        pts[:, 3] = np.array([0, self.height, 1], dtype=float)

        ref_pts = pts.copy()
        ref_pts = prj.apply_projection(self.H0, ref_pts)
        if self.H_gt is not None:
            P = matmul(self.H0, matmul(np.linalg.inv(H), matmul(np.linalg.inv(self.H0), matmul(self.H_gt, self.H0))))
            pts = prj.apply_projection(P, pts)

        vis = self.full_reference_image.copy()
        for i in range(4):
            pt1 = (int(ref_pts[0, i]), int(ref_pts[1, i]))
            pt2 = (int(ref_pts[0, (i+1) % 4]), int(ref_pts[1, (i+1) % 4]))
            vis = cv2.line(vis, pt1, pt2, (0, 255, 0), 3)

            if self.H_gt is not None:
                pt1 = (int(pts[0, i]), int(pts[1, i]))
                pt2 = (int(pts[0, (i+1) % 4]), int(pts[1, (i+1) % 4]))
                vis = cv2.line(vis, pt1, pt2, (255, 0, 255), 2)
        imvis.imshow(vis, title='Progress', wait_ms=10)

    def _update_warp(self, params, H):
        A = np.zeros((3, 3), dtype=float)
        for i in range(8):
            A += (params[i] * self.sl3_bases[i])
        G = np.zeros((3, 3), dtype=float)
        A_i = np.eye(3, dtype=float)
        factor_i = 1.0
        for i in range(9):
            G += (1.0 / factor_i) * A_i
            A_i = matmul(A_i, A)
            factor_i *= (i+1.0)

        if self.method == Method.IC:
            H = matmul(H, np.linalg.inv(G))
        elif self.method in [Method.FC, Method.ESM]:
            H = matmul(H, G)
        return H

    def _warp_current_image(self, img, H):
        res = cv2.warpPerspective(img, prj.matmul(self.H0, H), (self.width, self.height),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if self.verbose:
            imvis.imshow(res, 'Current Warp', wait_ms=10)
        return res


def _generate_warped_image(img, tx, ty, tz, rx, ry, rz):
    H = np.array([[0.93757391, -0.098535322, -8.3316984],
                  [0.0703476, 0.93736351, -32.40559],
                  [-4.9997212e-05, -4.9928687e-05, 1]], dtype=float)
    rows, cols = img.shape[:2]
    warped = cv2.warpPerspective(img, H, (cols, rows), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, H


def demo():
    img = imutils.imread('lenna.png')
    rect = (210, 200, 160, 183)
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
    pu.tic('FC')
    align = Alignment(target_template, Method.FC, full_reference_image=img, num_pyramid_levels=6, verbose=verbose)
    align.set_true_warp(H_gt)
    H_est, result = align.align(warped, H0)
    pu.toc('FC')
    imvis.imshow(result, 'Result FC', wait_ms=10)

    pu.tic('IC')
    align = Alignment(target_template, Method.IC, full_reference_image=img, num_pyramid_levels=4, verbose=verbose)
    align.set_true_warp(H_gt)
    H_est, result = align.align(warped, H0)
    pu.toc('IC')
    imvis.imshow(result, 'Result IC', wait_ms=10)

    pu.tic('ESM')
    align = Alignment(target_template, Method.ESM, full_reference_image=img, num_pyramid_levels=6, verbose=verbose)
    align.set_true_warp(H_gt)
    H_est, result = align.align(warped, H0)
    pu.toc('ESM')
    imvis.imshow(result, 'Result ESM', wait_ms=-1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
