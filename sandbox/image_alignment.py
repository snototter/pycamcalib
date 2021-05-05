import cv2
from enum import Enum
import logging
from vito import imutils, imvis, cam_projections as prj
from vito import pyutils as pu
import numpy as np

_logger = logging.getLogger('ImageAlignment')

# Python port of https://github.com/cashiwamochi/LK20_ImageAlignment

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


def _image_pyramid(src, num_levels):
    """Creates the Gaussian image pyramid (each level is upsampled to the original src image size)."""
    # pyrDown requires uint8 inputs
    down_sampled = src.copy()
    pyramid = list()
    # Convert to float (and range [0, 1])
    src = src.astype(float) / 255.0
    pyramid.append(src.copy())
    for l in range(num_levels - 1):
        down_sampled = cv2.pyrDown(down_sampled.copy())
        up_sampled = down_sampled.copy()
        for m in range(l+1):
            up_sampled = cv2.pyrUp(up_sampled.copy())
        up_sampled = up_sampled.astype(float) / 255.0
        pyramid.append(up_sampled)
    return pyramid


def _image_gradient(image):
    height, width = image.shape[:2]
    dx = np.column_stack((image[:, 1:] - image[:,:-1], np.zeros((height, 1), dtype=float)))
    dy = np.row_stack((image[1:,:] - image[:-1, :], np.zeros((1, width), dtype=float)))
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
    def __init__(self, image, method=Method.ESM, num_pyramid_levels=4, blur_kernel_size=(5, 5), max_iterations=50, full_reference_image=None):
        # TODO if verbose is True, you must provide the full reference image, too!
        self.verbose = False
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
        self.dxdy = list()  # TODO len == num_pyramid_levels
        self.J = list()  # 1 Jacobian per pyramid level   # TODO len == num_pyramid_levels
        self.H = list()  # 1 Hessian per pyramid level  TODO len == num_pyramid_levels
        self._precompute()

    def set_true_warp(self, H_gt):
        self.H_gt = H_gt.copy()
        if self.full_reference_image is None:
            _logger.error('To show the progress, you must provide the full reference image, too!')
            self.verbose = False
        else:
            self.verbose = True

    def track(self, image, H0):
        self.H0 = H0

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
        num_params = len(self.sl3_bases)
        Hessian = np.zeros((num_params, num_params), dtype=float)
        for r in range(J.shape[0]):
            row = J[r,:].reshape((1, -1))
            Hessian += matmul(np.transpose(row), row)
        return Hessian

    def _compute_Jacobian(self, dxdy, ref_dxdy):
        J = np.zeros((self.height*self.width, 8), dtype=float)
        if self.method == Method.ESM:
            assert ref_dxdy is not None
            Ji = (dxdy + ref_dxdy) / 2.0

        for u in range(self.height):
            for v in range(self.width):
                idx = u*self.width + v
                if self.method in [Method.FC, Method.IC]:
                    dd_row = dxdy[idx,:].reshape((1, -1))
                    J[idx,:] = matmul(dd_row, self.JwJg[idx])
                elif self.method == Method.ESM:
                    Ji_row = Ji[idx,:].reshape((1, -1))
                    J[idx,:] = matmul(Ji_row, self.JwJg[idx])
                else:
                    raise NotImplementedError()
        return J

    def _compute_JwJg(self):
        self.JwJg = list()
        for v in range(self.height):
            for u in range(self.width):
                # Eq.(63)
                Jw = np.array([
                               [u, v, 1, 0, 0, 0, -u*u, -u*v, -u],
                               [0, 0, 0, u, v, 1, -u*v, -v*v, -v]],
                              dtype=float)
                # Shapes: [2x8] = [2x9] * [9x8]
                JwJg = matmul(Jw, self.Jg)
                self.JwJg.append(JwJg)
    
    def _compute_residuals(self, cur_image, ref_image):
        res = 0.0
        residuals = np.zeros((self.height*self.width, 1), dtype=float)
        for v in range(self.height):
            for u in range(self.width):
                idx = v*self.width + u
                r = 0
                if self.method in [Method.IC or Method.ESM]:
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
                J_row = J[idx,:].reshape((1, -1))
                if self.method == Method.ESM:
                    params += -1 * np.transpose(J_row) * residuals[idx, 0]
                else:
                    params += np.transpose(J_row) * residuals[idx, 0]
        params = matmul(hessian_inv, params)
        return params

    def _is_converged(self, curr_error, prev_error):
        if prev_error < 0:
            return False
        if prev_error < curr_error + 1e-5:#0.0000001: # TODO check numerical stability
            return True
        return False
    
    def _precompute(self):
        if self.method == Method.FC:
            self._compute_JwJg()
        elif self.method == Method.IC:
            self._compute_JwJg()
            #TODO
# mvm_ref_DxDy.clear(); mvm_ref_DxDy.reserve(m_pyramid_level);
# mvm_J.clear(); mvm_J.reserve(m_pyramid_level);
# mvm_hessian.clear(); mvm_hessian.reserve(m_pyramid_level);

# for(int level = 0; level < m_pyramid_level; level++) {
#     cv::Mat m_dxdy = ComputeImageGradient(mvm_ref_image_pyramid[level]);
#     mvm_ref_DxDy.push_back(m_dxdy);
#     cv::Mat m_J = ComputeJ(m_dxdy); // JiJwJg
#     mvm_J.push_back(m_J);
#     cv::Mat m_hessian = ComputeHessian(m_J);
#     mvm_hessian.push_back(m_hessian);
# }
        elif self.method == Method.ESM:
            self._compute_JwJg()
            #TODO
# mvm_ref_DxDy.clear(); mvm_ref_DxDy.reserve(m_pyramid_level);
# for(int level = 0; level < m_pyramid_level; level++) {
#     cv::Mat m_dxdy = ComputeImageGradient(mvm_ref_image_pyramid[level]);
#     mvm_ref_DxDy.push_back(m_dxdy);
# }
        else:
            raise NotImplementedError()
        
    def _helper_process_fc(self, tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level):
        prev_error = -1
        curr_error = -1
        H = tmp_H.copy()
        for iteration in range(self.max_iterations):
            cur_working_image = self._warp_current_image(curr_image_pyramid, H)
            # imvis.imshow(cur_working_image, "warped current image", wait_ms=-1)
            residuals, curr_error = self._compute_residuals(cur_working_image, ref_image_pyramid)
            print(f'Iteration[{iteration:3d}] Level[{pyramid_level}]: {curr_error:.6f}')
            dxdy = _image_gradient(cur_working_image)
            J = self._compute_Jacobian(dxdy, None)
            Hessian = self._compute_Hessian(J)
            update_params = self._compute_update_params(Hessian, J, residuals)
            H_update = self._update_warp(update_params, H)

            if prev_error < 0:
                # Decide whether iterative update should be done in this pyramid level or not
                cur_working_image = self._warp_current_image(curr_image_pyramid, H_update)
                prev_error = curr_error
                residuals, curr_error = self._compute_residuals(cur_working_image, ref_image_pyramid)
                if self._is_converged(curr_error, prev_error):
                    break
                else:
                    H = H_update.copy()
                    if self.verbose:
                        self._show_progress(self.full_reference_image, H)
                    continue
            
            if self._is_converged(curr_error, prev_error):
                break
            else:
                prev_error = curr_error
                H = H_update.copy()
                if self.verbose:
                    self._show_progress(self.full_reference_image, H)
        return H, curr_error

    def _process_in_layer(self, tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level):
        if self.method == Method.FC:
            H, error = self._helper_process_fc(tmp_H, curr_image_pyramid, ref_image_pyramid, pyramid_level)
        elif self.method == Method.IC:
            raise NotImplementedError('TODO')
        elif self.method == Method.ESM:
            raise NotImplementedError('TODO')
        else:
            raise NotImplementedError()
        print(f'Method {self.method}, final residual: {error}')
        return H

    def _show_progress(self, canvas, H):
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

        vis = canvas.copy()
        for i in range(4):
            pt1 = (int(ref_pts[0, i]), int(ref_pts[1, i]))
            pt2 = (int(ref_pts[0, (i+1)%4]), int(ref_pts[1,(i+1)%4]))
            vis = cv2.line(vis, pt1, pt2, (0, 255, 0), 3)

            if self.H_gt is not None:
                pt1 = (int(pts[0, i]), int(pts[1, i]))
                pt2 = (int(pts[0, (i+1)%4]), int(pts[1,(i+1)%4]))
                vis = cv2.line(vis, pt1, pt2, (255, 0, 255), 2)
        imvis.imshow(vis, title='Progress', wait_ms=20)

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
        delta_H = G.copy()
        
        if self.method == Method.IC:
            H = matmul(H, np.linalg.inv(delta_H))
        elif self.method in [Method.FC, Method.ESM]:
            H = matmul(H, delta_H)
        return H
    
    def _warp_current_image(self, img, H):
        res = cv2.warpPerspective(img, prj.matmul(self.H0, H), (self.width, self.height),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return res


def _generate_warped_image(img, tx, ty, tz, rx, ry, rz):
    #TODO doesn't yield the same result as the Cpp simulator version
    # trans_x = 0.001 * tx
    # trans_y = 0.001 * ty
    # trans_z = 1 + 0.001 * tz
    # R_x = prj.rotx3d(rx*0.1/180.0*np.pi)
    # R_y = prj.rotx3d(ry*0.1/180.0*np.pi)
    # R_z = prj.rotx3d(rz*0.1/180.0*np.pi)
    # # Original code builds the matrix in ZYX (roll-pitch-yaw) order
    # R = matmul(R_z, matmul(R_y, R_x))
    # t = np.array([trans_x, trans_y, trans_z], dtype=float64).reshape((3,1))
    # rows, cols = img.shape[:2]
    # K = np.array([[1000, 0, cols/2],
    #               [0, 1000, rows/2],
    #               [0, 0, 1]], dtype=float64)
    # H = np.zeros_like(R)
    # H[0,0] = R[0,0]
    # H[1,0] = R[1,0]
    # H[2,0] = R[2,0]
    # H[0,1] = R[0,1]
    # H[1,1] = R[1,1]
    # H[2,1] = R[2,1]
    # H[0,2] = t[0,0]
    # H[1,2] = t[1,0]
    # H[2,2] = t[2,0]
    # H = matmul(K, matmul(H, np.linalg.inv(K)))
    # H /= H[2, 2]
    # _logger.info(f'Homography:\n{H}')
    H = np.array([[0.93757391, -0.098535322, -8.3316984],
                  [0.0703476, 0.93736351, -32.40559],
                  [-4.9997212e-05, -4.9928687e-05, 1]], dtype=float)
    rows, cols = img.shape[:2]
    warped = cv2.warpPerspective(img, H, (cols, rows), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, H


def demo():
    img = imutils.imread('lenna.png')
    rect = (210, 210, 160, 160) # TODO check with non-square rect
    target_template = imutils.roi(img, rect)
    # vis = img.copy()
    # vis = cv2.rectangle(vis, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 3)
    # imvis.imshow(vis, 'Input', wait_ms=10)
    imvis.imshow(target_template, 'Template', wait_ms=10)
    warped, H_gt = _generate_warped_image(img, -45, -25, 20, 30, -30, -360)
    imvis.imshow(warped, 'Simulated Warp', wait_ms=-1)
    
    # Initial estimate H0
    H0 = np.eye(3, dtype=float)
    H0[0, 2] = rect[0]
    H0[1, 2] = rect[1]
    _logger.info(f'Initial estimate, H0:\n{H0}')

    print('H0\n', H0)
    print('H_gt\n', H_gt)

    align = Alignment(target_template, Method.FC, full_reference_image=img, num_pyramid_levels=5)
    align.set_true_warp(H_gt)
    H_est, result = align.track(warped, H0)
    imvis.imshow(result, 'Result', wait_ms=-1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
