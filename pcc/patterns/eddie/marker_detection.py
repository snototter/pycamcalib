import cv2
import numpy as np
from dataclasses import dataclass
from vito import imutils, imvis
from ..common import GridIndex, Rect, Point


@dataclass
class ContourDetectionParams:
    marker_template_size_px: int = 64
    marker_margin_mm: int = 3

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
        
        # for C in ref_corners:
        #     cv2.circle(tpl_crop, (int(C.x), int(C.y)), 3, (255, 0, 0), 1)
        # imvis.imshow(tpl_full, 'tpl', wait_ms=10)
        # imvis.imshow(tpl_crop, 'cropped', wait_ms=-1)
        return {'template': tpl_crop, 'marker_corners': ref_corners}
    # #TODO exported SVG is 3-channel png!!

    # print('TPL CORNERS:', [c.to_int() for c in corners])
    # print('SORTED:', [c.to_int() for c in patterns.sort_points_ccw(corners)])
    # print('barycenter:', patterns.center(corners))
