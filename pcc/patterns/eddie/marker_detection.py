import cv2
import logging
import numpy as np
from dataclasses import dataclass, field
from vito import imutils, imvis, pyutils, cam_projections as pru
from ..common import GridIndex, Rect, Point, sort_points_ccw, points2numpy, image_corners, numpy2cvpt


@dataclass
class Transform:
    shape: dict = field(init=True, repr=False)
    homography : np.ndarray = field(init=True, repr=False)
    similarity : float
    rotation_deg : int
    marker_corners : list


@dataclass
class PreprocessingResult:
    original: np.ndarray = field(init=True, repr=False)
    gray: np.ndarray = field(init=True, repr=False)
    thresholded: np.ndarray = field(init=True, repr=False)
    edges: np.ndarray = field(init=True, repr=False)


@dataclass
class ContourDetectionParams:
    """
    The basic idea is to find rectangle-like shapes, warp the image to the
    reference view and verify via cross correlation.


    TODO doc members/params
    Configurable attributes:
        marker_template_size_px:    How large the reference template should be for
                                    correlation.

        marker_margin_mm:           Margin between central marker and border of the
                                    reference template for correlation

        simplification_factor:      To simplify the contours (Douglas-Peucker),
                                    we use an epsilon relative to the shape's
                                    arc length (perimeter).

        max_candidates_per_image:   Maximum number of candidate shapes to check per image

        # Canny edge detector parametrization
        edge_blur_kernel_size:      Size of the blur kernel for Canny edge detection, set
                                    to <= 0 to disable blurring.

        edge_canny_lower_thresh:    Lower hysteresis threshold for Canny edge detection.

        edge_canny_upper_thresh:    Upper hysteresis threshold for Canny edge detection.

        edge_sobel_aperture:        Sobel aperture size for Canny edge detection.

        edge_dilation_kernel_size:  To close small gaps after edge extraction, we use a
                                    dilation filter with this kernel size. Set to <= 0
                                    to disable dilation.
    """
    marker_template_size_px: int = 64
    marker_margin_mm: int = 3
    simplification_factor: float = 0.05
    max_candidates_per_image: int = 3 #FIXME
    edge_blur_kernel_size: int = 3
    edge_canny_lower_thresh: int = 50
    edge_canny_upper_thresh: int = 200
    edge_sobel_aperture: int = 3
    edge_dilation_kernel_size: int = 3

    # Acceptance threshold on the normalized correlation coefficient [-1, +1]
    marker_ccoeff_thresh: float = 0.9 #TODO doc
    marker_min_width_px: int = None #TODO doc
    grid_ccoeff_thresh: float = 0.8 #TODO doc
    debug: bool = True

    # def __post_init__(self):
    #     self._calibration_tpl = dict()
    # # # #TODO Note: SVG export is 3-channel png!

    @property
    def min_marker_area_px(self):
        if self.marker_min_width_px is None:
            return None
        return 0.7 * self.marker_min_width_px * self.marker_min_width_px


def _md_preprocess_img(img, det_params):
    """Image preprocessing: grayscale conversion, edge filtering, and
    some minor denoising operations."""
    gray = imutils.grayscale(img)
    #TODO     #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html    #findcontours should find (was???) White on black! - didn't look into it, but both b-on-w and w-on-b worked similarly well
    # Need to check why both thresh_bin and thresh_bin_inv works!
    _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Blur if needed
    if det_params.edge_blur_kernel_size > 0:
        bw = cv2.blur(bw.copy(), (det_params.edge_blur_kernel_size,
                                  det_params.edge_blur_kernel_size))
    # Edge detection
    edges = cv2.Canny(bw, det_params.edge_canny_lower_thresh,
                      det_params.edge_canny_upper_thresh,
                      apertureSize=det_params.edge_sobel_aperture)
    # Close minor gaps via dilation
    if det_params.edge_dilation_kernel_size > 0:
        kernel = np.ones((det_params.edge_dilation_kernel_size,
                          det_params.edge_dilation_kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return PreprocessingResult(original=img, gray=gray, thresholded=bw, edges=edges)


def _ensure_quadrilateral(shape):
    """Returns a 4-corner approximation of the given shape."""
    if shape['num_corners'] < 4 or shape['num_corners'] > 6:
        return None
    if shape['num_corners'] == 4:
        return shape
    #TODO is there a robust way to approximate a quad via line intersection?
    # what if the longest edge is not on the opposite side of the clipping image border/occluder?
    #
    # TODO as of now, this is just a nice-to-have functionality (lowest priority)
    #
    # # Find the longest edge
    # pts = [Point(x=pt[0, 0], y=pt[0, 1]) for pt in shape['hull']]
    # edges = [(pts[idx], pts[(idx+1) % len(pts)]) for idx in range(len(pts))]
    # edge_lengths = np.array([e[0].distance(e[1]) for e in edges])
    # idx_longest = np.argmax(edge_lengths)
    # # print('EDGE LENGTHS', edge_lengths, idx_longest)
    return None


def _md_find_shape_candidates(det_params, preprocessed, vis_img=None):
    """Locate candidate regions which could contain the marker."""
    # We don't want hierarchies of contours here, just the largest (i.e. the
    # root) contour of each hierarchy is fine:
    cnts = cv2.findContours(preprocessed.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Collect the convex hulls of all detected contours    
    shapes = list()
    for cnt in cnts:
        # Compute a simplified convex hull
        epsilon = det_params.simplification_factor*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Important: 
        # Simplification with too large epsilons could lead to intersecting
        # polygons. These would cause incorrect area computation, and more
        # "fun" behavior. Thus, we work with the shape's convex hull from
        # now on.
        hull = cv2.convexHull(approx)
        area = cv2.contourArea(hull)
        if det_params.min_marker_area_px is None or\
                area >= det_params.min_marker_area_px:
            shapes.append({'hull': hull, 'approx': approx, 'cnt': cnt,
                           'hull_area': area,
                           'num_corners': len(hull)})
    # Sort candidate shapes by area (descending)
    shapes.sort(key=lambda s: s['hull_area'], reverse=True)
    # Collect valid convex hulls, i.e. having 4-6 corners which could
    # correspond to a rectangular region.
    candidate_shapes = list()
    for shape in shapes:
        is_candidate = False
        if 3 < shape['num_corners'] < 6:
            candidate = _ensure_quadrilateral(shape)
            if candidate is not None:
                is_candidate = True
                candidate_shapes.append(candidate)
        if vis_img is not None:
            cv2.drawContours(vis_img, [shape['hull']], 0,
                             (0, 255, 0) if is_candidate else (255, 0, 0), 7)
        if det_params.max_candidates_per_image > 0 and det_params.max_candidates_per_image <= len(candidate_shapes):
            logging.info(f'Reached maximum amount of {det_params.max_candidates_per_image} candidate shapes.')
            break
    return candidate_shapes, vis_img

#FIXME REMOVE (rotation needs a reference point, e.g. image center)
# def _rotation(deg):
#     theta = np.deg2rad(deg)
#     ct = np.cos(theta)
#     st = np.sin(theta)
#     R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]], dtype=np.float32)
#     print (R)
#     return R
# def _concat_homography(T, H):
#     H = H / H[2, 2]
#     return pru.matmul(T, H)


def _find_transform(preprocessed, candidate, det_params, calibration_template):
    """Tries to find the homography between the calibration template and the given
    quadrilateral candidate."""
    # Ensure that both img and marker points are in the same (CCW) order
    img_corners = sort_points_ccw([Point(x=pt[0, 0], y=pt[0, 1]) for pt in candidate['hull']])
    coords_dst = points2numpy(calibration_template.refpts_cropped_marker)
    coords_src = points2numpy(img_corners)
    # Estimate homography
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return None
    warped = cv2.warpPerspective(preprocessed.thresholded, H,
                    (det_params.marker_template_size_px,
                     det_params.marker_template_size_px),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0))
    # Rotate the image 4 times and check if one orientation fits the template's center marker
    # This just takes 0.6-0.9ms in total (!), so no use in premature optimization.
    # Alternative ideas: sum reduction, profile comparison (e.g. via earth mover's distance or
    # even just L1), choose orientation with minimum "sum profile" difference.
    if det_params.debug:
        vis_img = calibration_template.tpl_cropped_marker.copy()
    transforms = [imutils.noop, imutils.rotate90,
                  imutils.rotate180, imutils.rotate270]
    similarities = np.zeros((len(transforms),))

    for idx, fx in zip(range(4), transforms):
        rotated = fx(warped)
        res = cv2.matchTemplate(rotated, calibration_template.tpl_cropped_marker, cv2.TM_CCOEFF_NORMED)
        similarities[idx] = res[0, 0]
        if det_params.debug:
            highlight_str = ' ***' if similarities[idx] > det_params.marker_ccoeff_thresh else ''
            print(f'orientation: {idx*90:3d}, similarity: {res.item(0):6.3f}{highlight_str}')
            vis_img = imutils.concat(vis_img, rotated, horizontal=True)
    best_idx = np.argmax(similarities)
    if similarities[best_idx] > det_params.marker_ccoeff_thresh:
        if det_params.debug:
            imvis.imshow(vis_img, 'Templated + Warped Candidate', wait_ms=100) #TODO remove
        first_idx = 3 - best_idx
        rotated_corners = [img_corners[(first_idx + i) % 4] for i in range(4)]
        return Transform(shape=candidate, homography=H,
                         similarity=similarities[best_idx],
                         rotation_deg=90*best_idx,
                         marker_corners=rotated_corners)
    return None


def _find_grid(preproc, transform, pattern_specs, det_params, vis=None):
    ctpl = pattern_specs.calibration_template  # Alias
    coords_dst = points2numpy(ctpl.refpts_full_marker)
    coords_src = points2numpy(transform.marker_corners)
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return vis
    h, w = ctpl.tpl_full.shape[:2]
    warped_img = cv2.warpPerspective(preproc.thresholded, H, (w, h), cv2.INTER_CUBIC)
    warped_mask = cv2.warpPerspective(np.ones(preproc.thresholded.shape[:2], dtype=np.uint8),
                                      H, (w, h), cv2.INTER_NEAREST)

    ncc = cv2.matchTemplate(warped_img, ctpl.tpl_cropped_circle, cv2.TM_CCOEFF_NORMED)  # mask must be template size??
    ncc[ncc < det_params.grid_ccoeff_thresh] = 0

    if det_params.debug:
        overlay = imutils.ensure_c3(imvis.overlay(ctpl.tpl_full, 0.3, warped_img, warped_mask))
        warped_img_corners = pru.apply_projection(H, points2numpy(image_corners(preproc.thresholded), Nx2=False))
        for i in range(4):
            pt1 = numpy2cvpt(warped_img_corners[:, i])
            pt2 = numpy2cvpt(warped_img_corners[:, (i+1)%4])
            cv2.line(overlay, pt1, pt2, color=(0, 0, 255), thickness=3)

        #FIXME FIXME FIXME
        # Idea: detect blobs in thresholded NCC
        # barycenter/centroid of each blob gives the top-left corner (then compute the relative offset to get the initial corner guess)
        tpl = ctpl.tpl_cropped_circle
        for niter in range(300):
            y, x = np.unravel_index(ncc.argmax(), ncc.shape)
            cv2.rectangle(overlay, (x, y), (x+tpl.shape[1], y+tpl.shape[0]), (255, 0, 255))
            left = x - tpl.shape[1] // 2
            right = left + tpl.shape[1]
            top = y - tpl.shape[0] // 2
            bottom = top + tpl.shape[0]
            ncc[top:bottom, left:right] = 0
            if niter % 10 == 0:
                imvis.imshow(imvis.pseudocolor(ncc, [-1, 1]), 'NCC Result', wait_ms=10)
                imvis.imshow(overlay, 'Projected image', wait_ms=10)

    if vis is not None:
        cv2.drawContours(vis, [transform.shape['hull']], 0, (200, 0, 200), 3)
    return vis

# improvement: sum reduce, chose best orientation
#TODO refactor (e.g. input images iterable, compute tpl once, ...)
#TODO what to return? ==> matching points
def find_target(img, pattern_specs, det_params=ContourDetectionParams()):
    # pyutils.tic('img-preprocessing')#TODO remove
    preprocessed = _md_preprocess_img(img, det_params)    
    if det_params.debug:
        from vito import imvis
        vis = imutils.ensure_c3(preprocessed.gray)
    # pyutils.toc('img-preprocessing')#TODO remove - 20-30ms
    # pyutils.tic('center-candidates-contours')#TODO remove
    candidate_shapes, vis = _md_find_shape_candidates(det_params, preprocessed, vis)
    # pyutils.toc('center-candidates-contours')#TODO remove 1-2ms
    # pyutils.tic('center-verification-projective')#TODO remove 1-2ms
    # Find best fitting candidate (if any)
    transforms = list()
    for shape in candidate_shapes:
        # Compute homography between marker template and detected candidate
        transform = _find_transform(preprocessed, shape, det_params, pattern_specs.calibration_template)
        if transform is not None:
            transforms.append(transform)
    transforms.sort(key=lambda t: t.similarity, reverse=True)
    # pyutils.toc('center-verification-projective')#TODO remove
    if len(transforms) > 0:
        _find_grid(preprocessed, transforms[0], pattern_specs, det_params, vis=vis)
    if det_params.debug:
        imvis.imshow(vis, title='contours', wait_ms=-1)
