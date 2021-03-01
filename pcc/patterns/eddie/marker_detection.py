import cv2
import logging
import numpy as np
from dataclasses import dataclass, field
from vito import imutils, imvis, pyutils, cam_projections as pru
from ..common import GridIndex, Rect, Point, sort_points_ccw, points2numpy

@dataclass
class CalibrationTemplate:
    tpl_img_full : np.ndarray = field(init=True, repr=False)
    marker_corners_full : list
    tpl_img_marker_crop : np.ndarray = field(init=True, repr=False)
    marker_corners_crop : list
    tpl_img_circle : np.ndarray = field(init=True, repr=False)


@dataclass
class Transform:
    shape: dict = field(init=True, repr=False)
    homography : np.ndarray = field(init=True, repr=False)
    similarity : float
    rotation_deg : int
    marker_corners : list


@dataclass
class PreprocResult:
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
    marker_ccoeff_thresh: float = 0.9

    _calibration_tpl: dict() = field(init=False, repr=False)  # Stores already computed calibration templates

    def __post_init__(self):
        self._calibration_tpl = dict()

    def get_template(self, pattern_specs):
        """Returns the marker template."""
        if pattern_specs.name not in self._calibration_tpl:
            # Relative position of the central marker
            marker_rect_relative, marker_rect_offset = \
                pattern_specs.get_relative_marker_rect(self.marker_margin_mm)
            # Crop the central marker and resize to the configured template size
            tpl_full = pattern_specs.render_image()
            tpl_h, tpl_w = tpl_full.shape[:2]
            tpl_roi = Rect(left=np.floor(tpl_w * marker_rect_relative.left),
                top=np.floor(tpl_h * marker_rect_relative.top),
                width=np.floor(tpl_w * marker_rect_relative.width),
                height=np.floor(tpl_h * marker_rect_relative.height))
            tpl_crop = cv2.resize(imutils.crop(tpl_full, tpl_roi.int_repr()),
                                dsize=(self.marker_template_size_px,
                                        self.marker_template_size_px),
                                interpolation=cv2.INTER_LANCZOS4)
            # Compute the reference corners for homography estimation
            ref_corners_crop = [
                        Point(x=marker_rect_offset.x*tpl_crop.shape[1],
                              y=marker_rect_offset.y*tpl_crop.shape[0]),
                        Point(x=tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
                              y=marker_rect_offset.y*tpl_crop.shape[0]),
                        Point(x=tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
                              y=tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0]),
                        Point(x=marker_rect_offset.x*tpl_crop.shape[1],
                              y=tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0])]
            # Ensure that they're in CCW order, w.r.t. to their barycenter
            ref_corners_crop = sort_points_ccw(ref_corners_crop)

            ### Compute the reference corners for the full template
            marker_rect_relative_full, _ = pattern_specs.get_relative_marker_rect(0)
            tpl_roi_full = Rect(left=np.floor(tpl_w * marker_rect_relative_full.left),
                top=np.floor(tpl_h * marker_rect_relative_full.top),
                width=np.floor(tpl_w * marker_rect_relative_full.width),
                height=np.floor(tpl_h * marker_rect_relative_full.height))
            # Compute the reference corners for image warping
            ref_corners_full = [tpl_roi_full.top_left, tpl_roi_full.bottom_left,
                           tpl_roi_full.bottom_right, tpl_roi_full.top_right]
            # Ensure that they're in CCW order, w.r.t. to their barycenter
            ref_corners_full = sort_points_ccw(ref_corners_full)


            #### Compute the circle template
            circ_rect_relative, circ_offset = pattern_specs.get_relative_marker_circle()
            tpl_roi_circ = Rect(left=np.floor(tpl_w * circ_rect_relative.left),
                top=np.floor(tpl_h * circ_rect_relative.top),
                width=np.floor(tpl_w * circ_rect_relative.width),
                height=np.floor(tpl_h * circ_rect_relative.height))
            tpl_crop_circ = imutils.crop(tpl_full, tpl_roi_circ.int_repr())
            imvis.imshow(tpl_crop_circ, "CIRCLE????", wait_ms=-1)
            # # Compute the reference corners for homography estimation
            # ref_corners_crop = [
            #             Point(x=marker_rect_offset.x*tpl_crop.shape[1],
            #                   y=marker_rect_offset.y*tpl_crop.shape[0]),
            #             Point(x=tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
            #                   y=marker_rect_offset.y*tpl_crop.shape[0]),
            #             Point(x=tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
            #                   y=tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0]),
            #             Point(x=marker_rect_offset.x*tpl_crop.shape[1],
            #                   y=tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0])]
            
            self._calibration_tpl[pattern_specs.name] = CalibrationTemplate(
                tpl_img_full=tpl_full, marker_corners_full=ref_corners_full,
                tpl_img_marker_crop=tpl_crop, marker_corners_crop=ref_corners_crop,
                tpl_img_circle=imutils.crop(tpl_full, tpl_roi.int_repr()))#tpl_img_circle=tpl_crop_circ) #FIXME FIXME FIXME
        return self._calibration_tpl[pattern_specs.name]
    # #TODO Note: SVG export is 3-channel png!


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
    return PreprocResult(original=img, gray=gray, thresholded=bw, edges=edges)
# TODO everything above should be refactored to a separate preproc module
# from ..preprocessing import ??
# or at least have a separate Preprocessing Class (must be configurable by a 
# UI widget later on....)


def _ensure_quadrilateral(shape):
    """Returns a 4-corner approximation of the given shape."""
    if shape['num_corners'] < 4 or shape['num_corners'] > 6:
        return None
    if shape['num_corners'] == 4:
        return shape
    print('TODO line intersection, approximation')
    return None


def _md_find_shape_candidates(det_params, edges, vis_img=None):
    """Locate candidate regions which could contain the marker."""
    # We don't want hierarchies of contours here, just the largest (i.e. the
    # root) contour of each hierarchy is fine:
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Collect the convex hulls of all detected contours    
    shapes = list()
    for cnt in cnts:
        # Compute a simplified convex hull
        epsilon = det_params.simplification_factor*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(approx)

        # Important: 
        # Simplification with too large epsilons could lead to intersecting
        # polygons. These would cause incorrect area computation, and more
        # "fun" behavior. Thus, we work with the shape's convex hull from
        # now on.
        shapes.append({'hull': hull, 'approx': approx, 'cnt': cnt,
                       'hull_area': cv2.contourArea(hull),
                       'num_corners': len(hull),
                       'rotation': None,
                       'similarity': None})
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
        # TODO v1 keep only the k largest shapes (strong perspective could cause circles closer
        # to the camera to be quite large!!) - v2 ratio test (if relative delta between subsequent
        # candidates is too small, abort)
        if det_params.max_candidates_per_image > 0 and det_params.max_candidates_per_image <= len(candidate_shapes):
            logging.info(f'Reached maximum amount of {det_params.max_candidates_per_image} candidate shapes.')
            break
    return candidate_shapes, vis_img


def _rotation(deg):
    theta = np.deg2rad(deg)
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]], dtype=np.float32)
    print (R)
    return R


# def _concat_homography(T, H):
#     H = H / H[2, 2]
#     return pru.matmul(T, H)


def _find_transform(img, candidate, det_params, calibration_template):
    """Tries to find the homography between the calibration template and the given
    quadrilateral candidate."""
    # Ensure that both img and marker points are in the same (CCW) order
    img_corners = sort_points_ccw([Point(x=pt[0, 0], y=pt[0, 1]) for pt in candidate['hull']])
    coords_dst = points2numpy(calibration_template.marker_corners_crop)
    coords_src = points2numpy(img_corners)
        
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return None
    warped = cv2.warpPerspective(img, H, (det_params.marker_template_size_px,
                                          det_params.marker_template_size_px)) #TODO border pad zero, replicate? (the latter)
    # Naive matching: try each possible rotation:
    vis_img = calibration_template.tpl_img_marker_crop.copy()
    pyutils.tic('naive')#TODO remove
    transforms = [imutils.noop, imutils.rotate90,
                  imutils.rotate180, imutils.rotate270]
    similarities = np.zeros((len(transforms),))
    
    for idx, fx in zip(range(4), transforms):
        rotated = fx(warped)
        res = cv2.matchTemplate(rotated, calibration_template.tpl_img_marker_crop, cv2.TM_CCOEFF_NORMED)
        similarities[idx] = res[0, 0]
        print(f'{idx:d}: {res.item(0)}')
        vis_img = imutils.concat(vis_img, rotated, horizontal=False)
    best_idx = np.argmax(similarities)
    pyutils.toc('naive')#TODO remove
    if similarities[best_idx] > det_params.marker_ccoeff_thresh:
        imvis.imshow(vis_img, 'TPL+WARPED', wait_ms=100) #TODO remove
        first_idx = 3 - best_idx
        rotated_corners = [img_corners[(first_idx + i) % 4] for i in range(4)]
        return Transform(shape=candidate, homography=H,
                         similarity=similarities[best_idx],
                         rotation_deg=90*best_idx,
                         marker_corners=rotated_corners)
    return None


def _find_grid(preproc, transform, calibration_template, pattern_specs, det_params, vis=None):
    coords_dst = points2numpy(calibration_template.marker_corners_full)
    coords_src = points2numpy(transform.marker_corners)
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return vis
    h, w = calibration_template.tpl_img_full.shape[:2]
    warped_img = cv2.warpPerspective(preproc.original, H, (w, h), cv2.INTER_CUBIC)
    warped_mask = cv2.warpPerspective(np.ones(preproc.original.shape[:2], dtype=np.uint8),
                                      H, (w, h), cv2.INTER_NEAREST)
    overlay = imvis.overlay(calibration_template.tpl_img_full, 0.3, warped_img, warped_mask) #FIXME add mask

    #TODO tpl = circle template #FIXME use circle template
    # ncc = cv2.matchTemplate(warped_img, calibration_template.tpl_img_circle, cv2.TM_CCOEFF_NORMED)  # mask must be template size??
    ncc = cv2.matchTemplate(warped_img, calibration_template.tpl_img_circle, cv2.TM_CCOEFF_NORMED)  # mask must be template size??

    # # print(ncc.shape, ncc.dtype, np.min(ncc[:]), np.max(ncc[:]), 'VS', warped_img.shape, 'VS TPL:', calibration_template.tpl_img_marker_crop.shape)
    # vis_crop=[0, 0, ncc.shape[1], ncc.shape[0]]
    # match_crop = imutils.crop(calibration_template.tpl_img_full, vis_crop)
    # # print(match_crop.shape, 'MASK:', warped_mask.shape, warped_mask.dtype)
    vis_ncc = np.zeros(warped_mask.shape)
    tpl = calibration_template.tpl_img_circle
    imvis.imshow(tpl, "CIRCLE TMPL", wait_ms=10)
    # vis_ncc[tpl.shape[0]-1:, tpl.shape[1]-1:] = ncc
    vis_ncc[:-tpl.shape[0]+1, :-tpl.shape[1]+1] = ncc
    # overlay = imvis.overlay(calibration_template.tpl_img_full, 0.1, 
    #     imvis.pseudocolor(vis_ncc), warped_mask) #FIXME add mask
    overlay = imvis.overlay(calibration_template.tpl_img_full, 0.3,
            warped_img, warped_mask)
    y,x = np.unravel_index(ncc.argmax(), ncc.shape)
    cv2.rectangle(overlay, (x, y), (x+tpl.shape[1], y+tpl.shape[0]), (255, 0, 255))
    imvis.imshow(overlay, 'Projected image', wait_ms=10)

#TODO calib_template: member template_marker, template_target
    if vis is not None:
        cv2.drawContours(vis, [transform.shape['hull']], 0, (200, 0, 200), 3)
    return vis

# improvement: sum reduce, chose best orientation
#TODO refactor (e.g. input images iterable, compute tpl once, ...)
#TODO what to return?
def find_marker(img, pattern_specs, det_params=ContourDetectionParams()):
    debug = True
    pyutils.tic('find_marker-preprocessing')#TODO remove
    preprocessed = _md_preprocess_img(img, det_params)    
    if debug:    
        from vito import imvis
        vis = imutils.ensure_c3(preprocessed.gray)
        # vis = imutils.ensure_c3(imvis.overlay(preprocessed.gray, 0.5, preprocessed.edges, preprocessed.edges))
    pyutils.toc('find_marker-preprocessing')#TODO remove
    pyutils.tic('find_marker-template')#TODO remove
    calibration_template = det_params.get_template(pattern_specs)
    pyutils.toc('find_marker-template')#TODO remove
    pyutils.tic('find_marker-contours')#TODO remove
    candidate_shapes, vis = _md_find_shape_candidates(det_params, preprocessed.edges, vis)
    pyutils.toc('find_marker-contours')#TODO remove
    pyutils.tic('find_marker-projective')#TODO remove
    # Find best fitting candidate (if any)
    transforms = list()
    for shape in candidate_shapes:
        # Compute homography between marker template and detected candidate
        transform = _find_transform(preprocessed.original, shape, det_params, calibration_template)
        if transform is not None:
            transforms.append(transform)
    transforms.sort(key=lambda t: t.similarity, reverse=True)
    pyutils.toc('find_marker-projective')#TODO remove
    if len(transforms) > 0:
        _find_grid(preprocessed, transforms[0], calibration_template, pattern_specs, det_params, vis=vis)
    # print('TPL CORNERS:', [c.int_repr() for c in corners])
    # print('SORTED:', [c.int_repr() for c in patterns.sort_points_ccw(corners)])
    # print('barycenter:', patterns.center(corners))
    imvis.imshow(vis, title='contours', wait_ms=-1)
