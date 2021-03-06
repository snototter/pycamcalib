import cv2
import sys
import logging
import numpy as np
from dataclasses import dataclass, field
from vito import imutils, imvis, pyutils, cam_projections as pru
from ..common import GridIndex, Rect, Point, Edge, sort_points_ccw, points2numpy, image_corners, numpy2cvpt, bottommost_point


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
    wb: np.ndarray = field(init=True, repr=False)  # white on black
    bw: np.ndarray = field(init=True, repr=False)  # black on white
    edges: np.ndarray = field(init=True, repr=False)


@dataclass
class CalibrationGridPoint:
    x: float
    y: float
    score: float


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
    simplification_factor: float = 0.01
    max_candidates_per_image: int = 10 #FIXME
    edge_blur_kernel_size: int = 3
    edge_canny_lower_thresh: int = 50
    edge_canny_upper_thresh: int = 200
    edge_sobel_aperture: int = 3
    edge_dilation_kernel_size: int = 3

    # Acceptance threshold on the normalized correlation coefficient [-1, +1]
    marker_ccoeff_thresh: float = 0.7 #TODO doc
    marker_min_width_px: int = None #TODO doc
    grid_ccoeff_thresh_initial: float = 0.6 #TODO doc
    grid_ccoeff_thresh_refine: float = 0.8
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
    wb = cv2.bitwise_not(bw)
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
    return PreprocessingResult(original=img, gray=gray, bw=bw, wb=wb, edges=edges)


def _ensure_quadrilateral(shape, img=None):
    """Returns a 4-corner approximation of the given shape."""
    if shape['num_corners'] < 4 or shape['num_corners'] > 8:
        return None
    if shape['num_corners'] == 4:
        return shape
    #TODO is there a robust way to approximate a quad via line intersection?
    # what if the longest edge is not on the opposite side of the clipping image border/occluder?
    #
    # TODO as of now, this is just a nice-to-have functionality (lowest priority)
    #
    # Assumption: the longest edge is fully visible - TODO verify if both endpoints are within
    # the image (not touching the border? but then there`could be an occluding object...) but such
    # bad examples wouldn't pass the NCC check anyhow....
    #
    # Find the longest edge
    pts = [Point(x=pt[0, 0], y=pt[0, 1]) for pt in shape['hull']]
    edges = [Edge(pts[idx], pts[(idx+1) % len(pts)]) for idx in range(len(pts))]
    edge_lengths = np.array([e.length for e in edges])
    idx_longest = np.argmax(edge_lengths)
    # Find most parallel edge
    most_parallel_angle = None
    idx_parallel = None
    orth_edges = list()
    for idx in range(len(edges)):
        ##### print('INTERSECTION DEMO: angle: ', edges[idx_longest].angle(edges[idx]), 'intersect:', edges[idx_longest].intersection(edges[idx]))
        if idx == idx_longest:
            continue
        angle = edges[idx_longest].angle(edges[idx])
        orth_edges.append((edges[idx], np.abs(angle - np.pi/2)))
        angle = np.abs(angle - np.pi)
        if most_parallel_angle is None or angle < most_parallel_angle:
            most_parallel_angle = angle
            idx_parallel = idx
        ##### print('Angle longest to {}: {} deg, idx: {}'.format(edges[idx], np.rad2deg(edges[idx_longest].angle(edges[idx])), idx_parallel))
    # Sort edges by "how orthogonal they are" w.r.t. to the longest edge
    orth_edges.sort(key=lambda oe: oe[1])
    # Remove the sorting key
    orth_edges = [oe[0] for oe in orth_edges]
    # Intersect the lines (longest & parallel with the two "most orthogonal") to
    # get the quad
    intersections = list()
    for pidx in [idx_longest, idx_parallel]:
        for oidx in [0, 1]:
            ip = edges[pidx].intersection(orth_edges[oidx])
            if ip is not None:
                intersections.append(ip)
    # Sort the intersection points in CCW order to get a convex hull, starting
    # from the bottommost point
    intersections = sort_points_ccw(intersections,
                                    pt_ref=bottommost_point(intersections))
    # Debug visualizations
    if img is not None:
        vis = img.copy()
        cv2.line(vis, edges[idx_longest].pt1.int_repr(), edges[idx_longest].pt2.int_repr(),
                 (255, 0, 0), 3)
        cv2.line(vis, edges[idx_parallel].pt1.int_repr(), edges[idx_parallel].pt2.int_repr(),
                 (255, 255, 0), 3)
        for idx in range(2):
            cv2.line(vis, orth_edges[idx].pt1.int_repr(), orth_edges[idx].pt2.int_repr(),
                     (0, 255, 255), 3)
        for ip in intersections:
            cv2.circle(vis, ip.int_repr(), 10, (255, 0, 255), 3)
        imvis.imshow(vis, 'Ensure quad: r=longest, y=parallel, c=orth, m=intersections', wait_ms=100)
    # Convert intersection points to same format as OpenCV uses for contours
    hull = np.zeros((len(intersections), 1, 2), dtype=np.int32)
    for idx in range(len(intersections)):
        hull[idx, 0, :] = intersections[idx].int_repr()
    shape['hull'] = hull
    shape['num_corners'] = len(intersections)
    # Unnecessary check for now - but we might change the quad approximation
    # later on, so for future safety, verify the number of hull points again:
    if shape['num_corners'] == 4:
        return shape
    return None


def _md_find_center_marker_candidates(det_params, preprocessed, vis_img=None):
    """Locate candidate regions which could contain the marker."""
    debug_shape_extraction = det_params.debug and True
    # We don't want hierarchies of contours here, just the largest (i.e. the
    # root) contour of each hierarchy is fine:
    cnts = cv2.findContours(preprocessed.wb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Collect the convex hulls of all detected contours    
    shapes = list()
    if debug_shape_extraction:
        tmp_vis = imutils.ensure_c3(preprocessed.wb)
        tmp_drawn = 0
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
        if debug_shape_extraction:
            cv2.drawContours(tmp_vis, [cnt], 0, (255, 0, 0), 7)
            cv2.drawContours(tmp_vis, [hull], 0, (255, 0, 255), 7)
            tmp_drawn += 1
            if tmp_drawn % 10 == 0:
                imvis.imshow(tmp_vis, 'Shape candidates', wait_ms=10)
        if det_params.min_marker_area_px is None or\
                area >= det_params.min_marker_area_px:
            shapes.append({'hull': hull, 'approx': approx, 'cnt': cnt,
                           'hull_area': area,
                           'num_corners': len(hull)})
    if debug_shape_extraction:
        print('Check "shape candidates". Press key to continue')
        imvis.imshow(tmp_vis, 'Shape candidates', wait_ms=-1)
    # Sort candidate shapes by area (descending)
    shapes.sort(key=lambda s: s['hull_area'], reverse=True)
    # Collect valid convex hulls, i.e. having 4-6 corners which could
    # correspond to a rectangular region.
    candidate_shapes = list()
    for shape in shapes:
        is_candidate = False
        if 3 < shape['num_corners'] <= 8:
            candidate = _ensure_quadrilateral(shape) #TODO pass image for debug visualizations, preprocessed.original)
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

#FIXME FIXME #In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.


def _find_center_marker_transform(preprocessed, candidate, det_params, calibration_template):
    """Estimates the homography between the calibration template and the given
    quadrilateral center marker candidate."""
    # Ensure that both the image and the marker points are in the same (CCW) order
    img_corners = sort_points_ccw([Point(x=pt[0, 0], y=pt[0, 1]) for pt in candidate['hull']])
    coords_dst = points2numpy(calibration_template.refpts_cropped_marker)
    coords_src = points2numpy(img_corners)
    # Estimate the homography
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return None
    warped = cv2.warpPerspective(preprocessed.bw, H,
                    (det_params.marker_template_size_px,
                     det_params.marker_template_size_px),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0))
    # Rotate the image 4 times by 90 degrees each and check if one orientation fits
    # the template's center marker sufficiently well.
    # This just takes 0.6-0.9ms in total (!), so there is no use for premature optimization.
    # Alternative ideas: sum reduction, profile comparison (e.g. via earth mover's distance or
    # even just L1), choose orientation with minimum "sum profile" difference.
    tpl_marker = calibration_template.tpl_cropped_marker
    if det_params.debug:
        vis_img = tpl_marker.copy()
    # Define the rotations we'll apply in the following loop
    transforms = [imutils.noop, imutils.rotate90,
                  imutils.rotate180, imutils.rotate270]
    similarities = np.zeros((len(transforms),))
    for idx, fx in zip(range(4), transforms):
        rotated = fx(warped)
        res = cv2.matchTemplate(rotated, tpl_marker, cv2.TM_CCOEFF_NORMED)
        similarities[idx] = res[0, 0]
        if det_params.debug:
            highlight_str = ' ***' if similarities[idx] > det_params.marker_ccoeff_thresh else ''
            print(f'orientation: {idx*90:3d}, similarity: {res.item(0):6.3f}{highlight_str}')
            vis_img = imutils.concat(vis_img, rotated, horizontal=True)
    # Find transformation which gave the highest similarity
    best_idx = np.argmax(similarities)
    if similarities[best_idx] > det_params.marker_ccoeff_thresh:
        # # if det_params.debug:
        # #     imvis.imshow(vis_img, 'Templated + Warped Candidate', wait_ms=100)
        # Re-order the image's center marker corners according
        # to the most fitting rotation:
        first_idx = 3 - best_idx
        rotated_corners = [img_corners[(first_idx + i) % 4] for i in range(4)]
        return Transform(shape=candidate, homography=H,
                         similarity=similarities[best_idx],
                         rotation_deg=90*best_idx,
                         marker_corners=rotated_corners)
    return None



def _find_initial_grid_points_contours(preproc, transform, pattern_specs, det_params, vis=None):
    print('WARNING - FINDING INITIAL GRID POINTS BY CONTOURS IS DEPRECATED')
    pyutils.tic('initial grid estimate - contour') #TODO remove
    ctpl = pattern_specs.calibration_template  # Alias
    coords_dst = points2numpy(ctpl.refpts_full_marker)
    coords_src = points2numpy(transform.marker_corners)
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return None, vis
    h, w = ctpl.tpl_full.shape[:2]
    # OpenCV doc: finding contours is finding white objects from black background!
    warped_img = cv2.warpPerspective(preproc.wb, H, (w, h), cv2.INTER_CUBIC)
    warped_mask = cv2.warpPerspective(np.ones(preproc.wb.shape[:2], dtype=np.uint8),
                                      H, (w, h), cv2.INTER_NEAREST)
    cnts = cv2.findContours(warped_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    vis_alt = imutils.ensure_c3(warped_img.copy())
    idx = 0
    expected_circle_area = (pattern_specs.calibration_template.dia_circle_px/2)**2 * np.pi
    exp_circ_area_lower = 0.5 * expected_circle_area
    exp_circ_area_upper = 2 * expected_circle_area
    for shape in cnts:
        area = cv2.contourArea(shape)
        if area < exp_circ_area_lower or area > exp_circ_area_upper:
            color=(255,0,0)
        else:
            color=(0, 0, 255)
            # continue
        # Centroid
        M = cv2.moments(shape)
        try:
            cx = np.round(M['m10']/M['m00'])
            cy = np.round(M['m01']/M['m00'])
        except ZeroDivisionError:
            continue
        
        idx += 1
        if det_params.debug:
            cv2.drawContours(vis_alt, [shape], 0, color, -1)
            cv2.circle(vis_alt, (int(cx), int(cy)), 1, (255, 255, 0), -1)
            if idx % 10 == 0:
                imvis.imshow(vis_alt, 'Points by contour', wait_ms=10)
    if det_params.debug:
        imvis.imshow(vis_alt, 'Points by contour', wait_ms=10)

    initial_estimates = list()
    #TODO match the points
    #TODO draw debug on vis
    pyutils.toc('initial grid estimate - contour') #TODO remove
    return initial_estimates, vis


def _find_initial_grid_points_correlation(preproc, transform, pattern_specs, det_params, vis=None):
    pyutils.tic('initial grid estimate - correlation') #TODO remove
    ctpl = pattern_specs.calibration_template  # Alias
    coords_dst = points2numpy(ctpl.refpts_full_marker)
    coords_src = points2numpy(transform.marker_corners)
    H = cv2.getPerspectiveTransform(coords_src, coords_dst)
    if H is None:
        return None, vis
    h, w = ctpl.tpl_full.shape[:2]
    warped_img = cv2.warpPerspective(preproc.bw, H, (w, h), cv2.INTER_CUBIC)
    warped_mask = cv2.warpPerspective(np.ones(preproc.bw.shape[:2], dtype=np.uint8),
                                      H, (w, h), cv2.INTER_NEAREST)

    ncc = cv2.matchTemplate(warped_img, ctpl.tpl_cropped_circle, cv2.TM_CCOEFF_NORMED)  # mask must be template size??
    ncc[ncc < det_params.grid_ccoeff_thresh_initial] = 0

    if det_params.debug:
        overlay = imutils.ensure_c3(imvis.overlay(ctpl.tpl_full, 0.3, warped_img, warped_mask))
        warped_img_corners = pru.apply_projection(H, points2numpy(image_corners(preproc.bw), Nx2=False))
        for i in range(4):
            pt1 = numpy2cvpt(warped_img_corners[:, i])
            pt2 = numpy2cvpt(warped_img_corners[:, (i+1)%4])
            cv2.line(overlay, pt1, pt2, color=(0, 0, 255), thickness=3)

    #FIXME FIXME FIXME
    # Idea: detect blobs in thresholded NCC
    # this could replace the greedy nms below
    # barycenter/centroid of each blob gives the top-left corner (then compute the relative offset to get the initial corner guess)
    initial_estimates = list()
    tpl = ctpl.tpl_cropped_circle
    while True:
        y, x = np.unravel_index(ncc.argmax(), ncc.shape)
        # print('Next', y, x, ncc[y, x], det_params.grid_ccoeff_thresh_initial, ncc.shape)
        if ncc[y, x] < det_params.grid_ccoeff_thresh_initial:
            break
        initial_estimates.append(CalibrationGridPoint(
            x=x, y=y, score=ncc[y, x]))
        # Clear the NCC peak around the detected template
        left = x - tpl.shape[1] // 2
        top = y - tpl.shape[0] // 2
        left, top, nms_w, nms_h = imutils.clip_rect_to_image(
                                    (left, top, tpl.shape[1], tpl.shape[0]),
                                    ncc.shape[1], ncc.shape[0])
        right = left + nms_w
        bottom = top + nms_h
        ncc[top:bottom, left:right] = 0
        if det_params.debug:
            cv2.rectangle(overlay, (x, y), 
                    (x+ctpl.tpl_cropped_circle.shape[1], y+ctpl.tpl_cropped_circle.shape[0]),
                    (255, 0, 255))
            if len(initial_estimates) % 20 == 0:
                # imvis.imshow(imvis.pseudocolor(ncc, [-1, 1]), 'NCC Result', wait_ms=10)
                imvis.imshow(overlay, 'Points by correlation', wait_ms=10)

    if vis is not None:
        cv2.drawContours(vis, [transform.shape['hull']], 0, (200, 0, 200), 3)
    if det_params.debug:
        print('Check "Points by correlation". Press key to continue')
        imvis.imshow(overlay, 'Points by correlation', wait_ms=-1)
    pyutils.toc('initial grid estimate - correlation') #TODO remove
    return initial_estimates, vis


def _find_grid(preproc, transform, pattern_specs, det_params, vis=None):
    # initial_estimates, vis = _find_initial_grid_points_contours(
    #         preproc, transform, pattern_specs, det_params, vis)
    initial_estimates, vis = _find_initial_grid_points_correlation(
            preproc, transform, pattern_specs, det_params, vis)
    #TODO refine!


# TODO example of pool.map for later parallelization: 
# https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
# https://stackoverflow.com/questions/659865/multiprocessing-sharing-a-large-read-only-object-between-processes

def sizeof_fmt(num, suffix='B'):
    # Taken from https://stackoverflow.com/a/1094933/400948
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
# improvement: sum reduce, chose best orientation
#TODO refactor (e.g. input images iterable, compute tpl once, ...)
#TODO what to return? ==> matching points
def find_target(img, pattern_specs, det_params=ContourDetectionParams()):
    # pyutils.tic('img-preprocessing')#TODO remove
    preprocessed = _md_preprocess_img(img, det_params)
    if det_params.debug:
        ### The pattern specification (+ rendered templates) takes ~1MB
        # # https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
        # print(f"""Object sizes:
        # pattern_spec: {sizeof_fmt(sys.getsizeof(pattern_specs))}
        # det_params:   {sizeof_fmt(sys.getsizeof(det_params))}
        # preprocessed: {sizeof_fmt(sys.getsizeof(preprocessed))}
        # """)
        # print('REQUIRES pympler!!')
        # from pympler import asizeof
        # print(f"""Sizes with pympler:
        # pattern_spec: {sizeof_fmt(asizeof.asizeof(pattern_specs))}
        # det_params:   {sizeof_fmt(asizeof.asizeof(det_params))}
        # preprocessed: {sizeof_fmt(asizeof.asizeof(preprocessed))}
        # """)
        from vito import imvis
        vis = imutils.ensure_c3(preprocessed.gray)
    else:
        vis = None
    # pyutils.toc('img-preprocessing')#TODO remove - 20-30ms
    # pyutils.tic('center-candidates-contours')#TODO remove
    candidate_shapes, vis = _md_find_center_marker_candidates(det_params, preprocessed, vis)
    # pyutils.toc('center-candidates-contours')#TODO remove 1-2ms
    # pyutils.tic('center-verification-projective')#TODO remove 1-2ms
    # Find best fitting candidate (if any)
    transforms = list()
    for shape in candidate_shapes:
        # Compute homography between marker template and detected candidate
        transform = _find_center_marker_transform(preprocessed, shape, det_params, pattern_specs.calibration_template)
        if transform is not None:
            transforms.append(transform)
    transforms.sort(key=lambda t: t.similarity, reverse=True)
    # pyutils.toc('center-verification-projective')#TODO remove
    if len(transforms) > 0:
        _find_grid(preprocessed, transforms[0], pattern_specs, det_params, vis=vis)
    # if det_params.debug:
    #     # print(f"""Object sizes after computation:
    #     # pattern_spec: {sizeof_fmt(sys.getsizeof(pattern_specs))}
    #     # det_params:   {sizeof_fmt(sys.getsizeof(det_params))}
    #     # preprocessed: {sizeof_fmt(sys.getsizeof(preprocessed))}
    #     # """)
    #     # print(f"""Sizes with pympler:
    #     # pattern_spec: {sizeof_fmt(asizeof.asizeof(pattern_specs))}
    #     # det_params:   {sizeof_fmt(asizeof.asizeof(det_params))}
    #     # preprocessed: {sizeof_fmt(asizeof.asizeof(preprocessed))}
    #     # """)
