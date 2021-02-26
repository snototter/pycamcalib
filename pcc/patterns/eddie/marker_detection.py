import cv2
import numpy as np
from dataclasses import dataclass
from vito import imutils, imvis, pyutils
from ..common import GridIndex, Rect, Point, sort_points_ccw


@dataclass
class ContourDetectionParams:
    # How large the reference template should be for correlation
    marker_template_size_px: int = 64

    # Margin between central marker and border of reference template for correlation
    marker_margin_mm: int = 3

    # RDP simplification of contours uses an epsilon relative to the shape's arc length/perimeter
    simplification_factor: float = 0.05

    def get_marker_template(self, pattern_specs):
        """Returns the marker template."""
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
        tpl_crop = cv2.resize(imutils.crop(tpl_full, tpl_roi.to_int()),
                              dsize=(self.marker_template_size_px,
                                     self.marker_template_size_px),
                              interpolation=cv2.INTER_LANCZOS4)
        # Compute the reference corners for homography estimation
        ref_corners = [Point(marker_rect_offset.x*tpl_crop.shape[1],
                             marker_rect_offset.y*tpl_crop.shape[0]),
                       Point(tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
                         marker_rect_offset.y*tpl_crop.shape[0]),
                       Point(tpl_crop.shape[1]-marker_rect_offset.x*tpl_crop.shape[1],
                             tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0]),
                       Point(marker_rect_offset.x*tpl_crop.shape[1],
                             tpl_crop.shape[0]-marker_rect_offset.y*tpl_crop.shape[0])]
        # Ensure that they're in CCW order, w.r.t. to their barycenter
        ref_corners = sort_points_ccw(ref_corners)
        
        # for C in ref_corners:
        #     cv2.circle(tpl_crop, (int(C.x), int(C.y)), 3, (255, 0, 0), 1)
        # imvis.imshow(tpl_full, 'tpl', wait_ms=10)
        # imvis.imshow(tpl_crop, 'cropped', wait_ms=-1)
        return {'template': tpl_crop, 'marker_corners': ref_corners}
    # #TODO Note: SVG export is 3-channel png!



#TODO refactor (e.g. input images iterable, compute tpl once, ...)
#TODO what to return?
def find_marker(img, pattern_specs, det_params=ContourDetectionParams()):
    pyutils.tic('preprocessing')
    #TODO separate preprocessing (common (parent) module!)
    gray = imutils.grayscale(img)
    #TODO     #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html    #findcontours should find (was???) White on black! - didn't look into it, but both b-on-w and w-on-b worked similarly well
    _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bw = cv2.blur(bw.copy(), (3,3))
    edges = cv2.Canny(bw, 50, 200, apertureSize=3)
    # Close minor gaps via dilation
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    from vito import imvis
    vis = imutils.ensure_c3(imvis.overlay(edges, gray, 0.8, edges))
    # vis = imutils.ensure_c3(edges)
    
    # TODO everything above should be refactored to a separate preproc module
    # from ..preprocessing import ??
    # or at least have a separate Preprocessing Class (must be configurable by a 
    # UI widget later on....)
    pyutils.toc('preprocessing')
    pyutils.tic('contours')

    # The basic idea is to find rectangle-like shapes, warp the image to the
    # reference view and verify via cross correlation.
    # We don't want hierarchies of contours here, just the largest (i.e. the
    # root) contour of each hierarchy is fine:
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

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
        shapes.append({'hull': hull, 'approx': approx,
                       'hull_area': cv2.contourArea(hull),
                       'num_corners': len(hull)})
    
    pyutils.toc('contours')
    pyutils.tic('projective')
    # Sort candidate shapes by area (descending)
    shapes.sort(key=lambda s: s['hull_area'], reverse=True)
    #TODO 
    marker_template = det_params.get_marker_template(pattern_specs)
    num_candidates = 0
    for shape in shapes:
        cv2.drawContours(vis, [shape['hull']], 0,
                         (0, 255, 0) if 3 < shape['num_corners'] < 6 else (255, 0, 0), 3)
        num_candidates += 1
        # TODO v1 keep only the k largest shapes (strong perspective could cause circles closer
        # to the camera to be quite large!!) - v2 ratio test (if relative delta between subsequent
        # candidates is too small, abort)

        #TODO if num_corners == 4 -> good, else try to simplify more (line fitting? take longest edges as reference, ....)
        #TODO sort_points_ccw(corner_pts)
        #TODO estimate perspective transform
        #TODO warp & correlate
        # if i < 15:
            # print('Largest', i, approx['approx'], approx['area'])
            # imvis.imshow(vis, title='contours', wait_ms=-1)
    pyutils.toc('projective')
    # print('TPL CORNERS:', [c.to_int() for c in corners])
    # print('SORTED:', [c.to_int() for c in patterns.sort_points_ccw(corners)])
    # print('barycenter:', patterns.center(corners))

    imvis.imshow(vis, title='contours', wait_ms=-1)
