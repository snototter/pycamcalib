import cv2
import logging
import numpy as np
from dataclasses import dataclass, field
from vito import imutils, imvis, pyutils
from ..common import GridIndex, Rect, Point, sort_points_ccw, points2numpy

@dataclass
class MarkerTemplate:
    template_img : np.ndarray = field(init=True, repr=False)
    marker_corners : list


@dataclass
class Transform:
    shape: dict = field(init=True, repr=False)
    homography : np.ndarray = field(init=True, repr=False)
    similarity : float
    rotation_deg : int


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


    _marker_tpl: dict() = field(init=False, repr=False)  # Stores already computed marker templates

    def __post_init__(self):
        self._marker_tpl = dict()

    def get_marker_template(self, pattern_specs):
        """Returns the marker template."""
        if pattern_specs.name not in self._marker_tpl:
            # Relative position of central marker
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
            ref_corners = [
                        Point(x=marker_rect_offset.x*tpl_crop.shape[1],
                              y=marker_rect_offset.y*tpl_crop.shape[0]),
                        Point(x=tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
                              y=marker_rect_offset.y*tpl_crop.shape[0]),
                        Point(x=tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
                              y=tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0]),
                        Point(x=marker_rect_offset.x*tpl_crop.shape[1],
                              y=tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0])]
            # Ensure that they're in CCW order, w.r.t. to their barycenter
            ref_corners = sort_points_ccw(ref_corners)
            
            # for C in ref_corners:
            #     cv2.circle(tpl_crop, (int(C.x), int(C.y)), 3, (255, 0, 0), 1)
            # imvis.imshow(tpl_full, 'tpl', wait_ms=10)
            # imvis.imshow(tpl_crop, 'cropped', wait_ms=-1)
            # mtpl = {'template': tpl_crop, 'marker_corners': ref_corners}
            # mtpl = {'template': tpl_crop, 'marker_corners': ref_corners}
            self._marker_tpl[pattern_specs.name] = MarkerTemplate(
                template_img=tpl_crop, marker_corners=ref_corners)
        return self._marker_tpl[pattern_specs.name]
    # #TODO Note: SVG export is 3-channel png!


def _md_preprocess_img(img, det_params):
    """Image preprocessing: grayscale conversion, edge filtering, and
    some minor denoising operations."""
    gray = imutils.grayscale(img)
    #TODO     #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html    #findcontours should find (was???) White on black! - didn't look into it, but both b-on-w and w-on-b worked similarly well
    # Need to check why both thresh_bin and thresh_bin_inv works!
    _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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
    return gray, edges
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


def _md_find_shape_candidates(det_params, marker_template, edges, vis_img=None):
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


def _find_transform(img, candidate, det_params, marker_template):
    """Tries to find the homography between the marker template and the given
    quadrilateral candidate."""
    # Ensure that both img and marker points are in the same (CCW) order
    img_corners = sort_points_ccw([Point(x=pt[0, 0], y=pt[0, 1]) for pt in candidate['hull']])
    coords_dst = points2numpy(marker_template.marker_corners)
    coords_src = points2numpy(img_corners)
        
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return None
    warped = cv2.warpPerspective(img, H, (det_params.marker_template_size_px,
                                          det_params.marker_template_size_px)) #TODO border pad zero, replicate? (the latter)
    # Naive matching: try each possible rotation:
    vis_img = marker_template.template_img.copy()
    pyutils.tic('naive')#TODO remove
    transforms = [imutils.noop, imutils.rotate90,
                  imutils.rotate180, imutils.rotate270]
    similarities = np.zeros((len(transforms),))
    for idx, fx in zip(range(4), transforms):
        rotated = fx(warped)
        res = cv2.matchTemplate(rotated, marker_template.template_img, cv2.TM_CCOEFF_NORMED)
        similarities[idx] = res[0, 0]
        print(f'{idx:d}: {res.item(0)}')
        vis_img = imutils.concat(vis_img, rotated, horizontal=False)
    best_idx = np.argmax(similarities)
    pyutils.toc('naive')#TODO remove
    imvis.imshow(vis_img, 'TPL+WARPED', wait_ms=300) #TODO remove
    if similarities[best_idx] > det_params.marker_ccoeff_thresh:
        return Transform(shape=candidate, homography=H,
                         similarity=similarities[best_idx],
                         rotation_deg=90*best_idx)
    return None


# improvement: sum reduce, chose best orientation
#TODO refactor (e.g. input images iterable, compute tpl once, ...)
#TODO what to return?
def find_marker(img, pattern_specs, det_params=ContourDetectionParams()):
    debug = True
    pyutils.tic('find_marker-preprocessing')#TODO remove
    gray, edges = _md_preprocess_img(img, det_params)    
    if debug:    
        from vito import imvis
        vis = imutils.ensure_c3(imvis.overlay(gray, 0.5, edges, edges))
    pyutils.toc('find_marker-preprocessing')#TODO remove
    pyutils.tic('find_marker-template')#TODO remove
    marker_template = det_params.get_marker_template(pattern_specs)
    pyutils.toc('find_marker-template')#TODO remove
    pyutils.tic('find_marker-contours')#TODO remove
    candidate_shapes, vis = _md_find_shape_candidates(det_params, marker_template, edges, vis)
    pyutils.toc('find_marker-contours')#TODO remove
    pyutils.tic('find_marker-projective')#TODO remove
    # Find best fitting candidate (if any)
    transforms = list()
    for shape in candidate_shapes:
        # Compute homography between marker template and detected candidate
        transform = _find_transform(img, shape, det_params, marker_template)
        if transform is not None:
            transforms.append(transform)
    transforms.sort(key=lambda t: t.similarity, reverse=True)
    pyutils.toc('find_marker-projective')#TODO remove
    if len(transforms) > 0:
        if vis is not None:
            cv2.drawContours(vis, [transforms[0].shape['hull']], 0, (200, 0, 200), 3)
    # print('TPL CORNERS:', [c.int_repr() for c in corners])
    # print('SORTED:', [c.int_repr() for c in patterns.sort_points_ccw(corners)])
    # print('barycenter:', patterns.center(corners))
    imvis.imshow(vis, title='contours', wait_ms=-1)
