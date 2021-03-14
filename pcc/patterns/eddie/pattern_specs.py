import logging
import io
import svgwrite
import cv2
import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from dataclasses import dataclass, field
from vito import imutils, imvis
from collections import deque
from ..common import GridIndex, Rect, Point, sort_points_ccw, center, SpecificationError

#TODO add member for default file location (within the package once we deploy it)
# this would allow loading svg/png w/o rendering first...

@dataclass
class CalibrationTemplate:
    tpl_full: np.ndarray = field(init=True, repr=False)
    refpts_full_marker: list
    tpl_cropped_marker: np.ndarray = field(init=True, repr=False)
    refpts_cropped_marker: list
    tpl_cropped_circle: np.ndarray = field(init=True, repr=False)
    refpts_cropped_circle: list
    dia_circle_px: float


@dataclass
class ReferencePoint:
    col: int
    row: int
    pos_mm_tl: Point  # Pos in mm relative to the target's top left corner
    pos_mm_centered: Point # Pos in mm relative to the reference grid
    surrounding_circles: list


@dataclass(frozen=True)
class PatternSpecificationEddie:
    """Specification of an Eddie-like calibration pattern as
    we typically use at work.

    Configurable attributes:
        name:               Identifier of this calibration pattern

        target_width_mm:    Width of the calibration target (not the
                            printable/grid area) in [mm]

        target_height_mm:   Height of the calibration target in [mm]

        dia_circles_mm:     Circle diameter in [mm]

        dist_circles_mm:    Center distance between neighboring circles in [mm]

        circles_per_square_edge_length:     Number of circles along each
                                            edge of the square

        circles_per_square_edge_thickness:  Thickness of the square border,
                                            given as a number of circles

        min_margin_target_mm:   Minimum distance between edge of the target
                                and circles, will be adjusted automatically
                                to avoid cropping circles (refer to the
                                "computed_margins" property)

        bg_color:   SVG color used to fill the background, specify via:
                    * named colors: white, red, orange, ...
                    * hex colors: #ff9e2c
                    * rgb colors: rgb(255, 128, 44)

        fg_color:   SVG color used to draw the foreground (circles and
                    center marker), see bg_color

    Calculated attributes:
        r_circles_mm:           Radius of a circle

        circles_per_row:        Number of circles along each row

        circles_per_col:        Number of circles along each column

        skipped_circle_indices: Grid indices which should be skipped
                                because they're occluded by the central
                                marker

        square_topleft_corner:  Grid indices of the top left corner of
                                the central marker

        marker_size_mm:         Length of the center marker's edge in [mm].

        marker_thickness_mm     Thickness of the center marker's border in [mm].
    """
    # Attributes to be set by the user
    name: str
    target_width_mm: int
    target_height_mm: int
    dia_circles_mm: int
    dist_circles_mm: int
    circles_per_square_edge_length: int = 4
    circles_per_square_edge_thickness: int = 1
    min_margin_target_mm: int = 5
    bg_color: str = 'white'
    fg_color: str = 'black'
    template_marker_size_px: int = 64 # FIXME doc
    template_marker_margin_mm: int = 3 # FIXME doc


    # Automatically calculated attributes
    r_circles_mm : float = field(init=False, repr=False)
    circles_per_row : int = field(init=False, repr=False)
    circles_per_col : int = field(init=False, repr=False)
    skipped_circle_indices: list = field(init=False, repr=False)
    square_topleft_corner: GridIndex = field(init=False, repr=False)
    marker_size_mm: int = field(init=False, repr=False)
    marker_thickness_mm: int = field(init=False, repr=False)
    calibration_template: CalibrationTemplate = field(init=False, repr=False)
    reference_points: list = field(init=False, repr=False)

    def __post_init__(self):
        # If the dataclass is frozen, we cannot set the computed values directly (https://stackoverflow.com/a/54119384/400948)
        object.__setattr__(self, 'r_circles_mm', self.dia_circles_mm/2)
        object.__setattr__(self, 'circles_per_row', ((self.target_width_mm - 2*self.min_margin_target_mm - self.dia_circles_mm) // self.dist_circles_mm) + 1)
        object.__setattr__(self, 'circles_per_col', ((self.target_height_mm - 2*self.min_margin_target_mm - self.dia_circles_mm) // self.dist_circles_mm) + 1)
        object.__setattr__(self, 'marker_size_mm', (self.circles_per_square_edge_length - 1) * self.dist_circles_mm + self.dia_circles_mm)
        object.__setattr__(self, 'marker_thickness_mm', (self.circles_per_square_edge_thickness - 1) * self.dist_circles_mm + self.dia_circles_mm)
        self._compute_square_topleft()
        self._compute_skipped_circles()
        self._verify()
        self._compute_calibration_template()
        self._compute_reference_grid()

    def _verify(self):
        # Circles mustn't touch
        if self.r_circles_mm >= self.dist_circles_mm / 2:
            raise SpecificationError('Circles mustn''t touch!')
        # Square marker must contain at least 1 row holding 2 circles
        # to estimate orientation
        if (self.circles_per_square_edge_length - 2*self.circles_per_square_edge_thickness < 2)\
                or (self.circles_per_square_edge_length//2 - self.circles_per_square_edge_thickness <= 0):
            raise SpecificationError('The center marker must contain at least 1 row with 2 circles for orientation!')
        if self.circles_per_square_edge_length % 2 == 1:
            raise SpecificationError('The number of circles along the center marker''s edge must be even!')

    def _compute_square_topleft(self):
        gw = self.circles_per_row
        gh = self.circles_per_col
        nx = gw - self.circles_per_square_edge_length
        if nx % 2 == 1:
            logging.warning('Square center marker won''t be centered horizontally.')
        left = nx // 2
        ny = gh - self.circles_per_square_edge_length
        if ny % 2 == 1:
            logging.warning('Square center marker won''t be centered vertically.')
        top = ny // 2
        object.__setattr__(self, 'square_topleft_corner', GridIndex(row=top, col=left))
        # cx = left + self.circles_per_square_edge_length // 2
        # cy = top + self.circles_per_square_edge_length // 2
        # print('TODO eddie center: ', cx, cy, ' grid dimension: ', gw, gh)

    def _compute_skipped_circles(self):
        sidx = list()
        for ridx in range(self.circles_per_square_edge_length):
            for cidx in range(self.circles_per_square_edge_length):
                keep = (ridx >= self.circles_per_square_edge_thickness) and\
                        (ridx < self.circles_per_square_edge_length//2) and\
                        (cidx >= self.circles_per_square_edge_thickness) and\
                        (cidx < self.circles_per_square_edge_length - self.circles_per_square_edge_thickness)
                if not keep:
                    sidx.append(GridIndex(row=self.square_topleft_corner.row+ridx,
                                          col=self.square_topleft_corner.col+cidx))
        object.__setattr__(self, 'skipped_circle_indices', sidx)
    
    @property
    def computed_margins(self):
        """
        Returns the actual margins between the target border and
        the first foreground ink/pixels, depending on the pattern's
        specification.
        """
        free_x = self.target_width_mm - (self.circles_per_row - 1) * self.dist_circles_mm - self.dia_circles_mm
        free_y = self.target_height_mm - (self.circles_per_col - 1) * self.dist_circles_mm - self.dia_circles_mm
        return free_x/2, free_y/2
    
    @property
    def _square_rects(self):
        tl = self.square_topleft_corner
        mx, my = self.computed_margins
        offset_x = mx + self.r_circles_mm
        offset_y = my + self.r_circles_mm

        rects = list()
        # Top border
        top = tl.row * self.dist_circles_mm + offset_y - self.r_circles_mm
        left = tl.col * self.dist_circles_mm + offset_x - self.r_circles_mm
        rects.append(Rect(left=left, top=top, width=self.marker_size_mm, height=self.marker_thickness_mm))
        # Left border
        rects.append(Rect(left=left, top=top, width=self.marker_thickness_mm, height=self.marker_size_mm))
        # Right
        x = left + self.marker_size_mm - self.marker_thickness_mm
        rects.append(Rect(left=x, top=top, width=self.marker_thickness_mm, height=self.marker_size_mm))
        # Bottom half
        length_half = self.marker_size_mm / 2
        rects.append(Rect(left=left, top=top+length_half, width=self.marker_size_mm, height=length_half))
        return rects

    def _skip_circle_idx(self, row, col):
        """
        Returns True if there should be no circle at the current
        grid position (row/col index).
        """
        for skip in self.skipped_circle_indices:
            if row == skip.row and col == skip.col:
                return True
        return False

    def __str__(self):
        mx, my = self.computed_margins
        return f"""[{self.name}]
* Target size: {self.target_width_mm}mm x {self.target_height_mm}mm, margins: {mx}mm, {my}mm
* Circles: d={self.dia_circles_mm}mm, distance={self.dist_circles_mm}mm
* Grid: {self.circles_per_row} x {self.circles_per_col}
* Marker: {self.marker_size_mm}mm x {self.marker_size_mm}mm, border: {self.marker_thickness_mm}mm
* Colors: {self.fg_color} on {self.bg_color}"""
#TODO add remaining specs to __str__

    def render_svg(self):
        logging.info(f'Drawing calibration pattern: {self}')
        h_mm = self.target_height_mm
        w_mm = self.target_width_mm

        # Helper to put fully specified coordinates (in millimeters)
        def _mm(v):
            return f'{v}mm'

        dwg = svgwrite.Drawing(profile='full')#, viewbox='0 0')
    #, height=f'{h_target_mm}mm', width=f'{w_target_mm}mm', profile='tiny', debug=False)
        # Height/width weren't set properly in the c'tor (my SVGs had 100% instead
        # of the desired dimensions). Thus, set the attributes manually:
        dwg.attribs['height'] = _mm(h_mm)
        dwg.attribs['width'] = _mm(w_mm)

        dwg.defs.add(dwg.style(f".grid {{ stroke: {self.fg_color}; stroke-width:1px; }}"))

        # Background should not be transparent
        dwg.add(dwg.rect(insert=(0, 0), size=(_mm(w_mm), _mm(h_mm)), fill=self.bg_color))

        # Draw circles
        margin_x, margin_y = self.computed_margins
        offset_x = margin_x + self.r_circles_mm
        offset_y = margin_y + self.r_circles_mm

        grid = dwg.add(dwg.g(id='circles'))
        cy_mm = offset_y
        for ridx in range(self.circles_per_col):
            cx_mm = offset_x
            for cidx in range(self.circles_per_row):
                if not self._skip_circle_idx(ridx, cidx):
                    grid.add(dwg.circle(center=(_mm(cx_mm), _mm(cy_mm)),
                        r=_mm(self.r_circles_mm), fill=self.fg_color))
                cx_mm += self.dist_circles_mm
            cy_mm += self.dist_circles_mm
        # Draw Eddie's face
        marker = dwg.add(dwg.g(id='marker'))
        for r in self._square_rects:
            marker.add(dwg.rect(insert=(_mm(r.left), _mm(r.top)),
                                size=(_mm(r.width), _mm(r.height)),
                                class_="grid"))#fill=pspecs.bg_color)
        #FIXME remove
        # # Add a dummy rect to check the printed area
        # dwg.add(dwg.rect(insert=(_mm(pspecs.min_margin_target_mm), _mm(pspecs.min_margin_target_mm)),
        #                  size=(_mm(w_mm - 2*pspecs.min_margin_target_mm), _mm(h_mm - 2*pspecs.min_margin_target_mm)),
        #                  stroke='black', stroke_width='1mm', fill='none'))
        return dwg

    def export_svg(self, filename):
        dwg = self.render_svg()
        dwg.saveas(filename, pretty=True)

    def render_image(self):
        """Renders the calibration pattern to an image (NumPy ndarray),
        performing all actions in-memory."""
        # Load the rendered SVG into a StringIO
        svg_sio = io.StringIO(self.render_svg().tostring())
        # Render it to PNG in-memory
        dwg_input = svg2rlg(svg_sio)
        img_mem_file = io.BytesIO()
        renderPM.drawToFile(dwg_input, img_mem_file, fmt="PNG")  # NEVER EVER SET DPI! doesn't scale properly as of 2021-03
        return imutils.memory_file2ndarray(img_mem_file)

    def _get_relative_marker_circle_v1(self):
        #TODO
        #return an arbitrary circle template
        #### V1 single circle
        gidx = GridIndex(1, 1)
        mx, my = self.computed_margins
        # Choose (almost) any template size (increase if you want more circles, decrease at will)
        tpl_size_mm = 2 * self.dia_circles_mm  

        cx = mx + self.r_circles_mm + gidx.col * self.dist_circles_mm
        cy = my + self.r_circles_mm + gidx.row * self.dist_circles_mm

        left = cx - tpl_size_mm / 2
        top = cy - tpl_size_mm / 2

        rect = Rect(left=left / self.target_width_mm, top=top / self.target_height_mm,
                    width=tpl_size_mm / self.target_width_mm, height=tpl_size_mm / self.target_height_mm)
        tpl_ref_pts = [Point(x=(cx - left) / tpl_size_mm, y=(cy - top) / tpl_size_mm)]
        return rect, tpl_ref_pts
    
    def _get_relative_marker_circle_v2(self):
        #TODO
        #return an arbitrary circle template
        #### V2 4 circles (yielding 5 reference points would lead to highly redundant matches - only ref center for now!!!)
        gidx = GridIndex(1.5, 1.5)
        mx, my = self.computed_margins

        # Offsets to the first circle's center:
        first_center_offset_x = mx + self.r_circles_mm
        first_center_offset_y = my + self.r_circles_mm

        padding_mm = 2#self.dist_circles_mm / 2
        # # tpl_size_mm = 2 * self.dia_circles_mm  # Choose (almost) any template size (increase if you want more circles, decrease at will)

        cx = first_center_offset_x + gidx.col * self.dist_circles_mm
        cy = first_center_offset_y + gidx.row * self.dist_circles_mm
        
        tpl_size_mm = 2*padding_mm + self.dia_circles_mm + self.dist_circles_mm
        left = cx - tpl_size_mm / 2
        top = cy - tpl_size_mm / 2
        
        rect = Rect(left=left / self.target_width_mm, top=top / self.target_height_mm,
                    width=tpl_size_mm / self.target_width_mm, height=tpl_size_mm / self.target_height_mm)
        tpl_ref_pts = [Point(x=0.5, y=0.5)]
        return rect, tpl_ref_pts


    def _get_relative_marker_rect(self, margin_mm):
        """
        Returns the center marker's position as a fraction of the calibration
        target's dimensions.
        Useful to extract the marker template for localization - because
        adjusting the SVG viewbox didn't work well for exporting cropped PNGs.

        :margin_mm: Specify the margin in [mm]
        """
        tlidx = self.square_topleft_corner
        mx, my = self.computed_margins
        offset_x = mx + self.r_circles_mm
        offset_y = my + self.r_circles_mm

        top = tlidx.row * self.dist_circles_mm + offset_y - self.r_circles_mm - margin_mm
        left = tlidx.col * self.dist_circles_mm + offset_x - self.r_circles_mm - margin_mm
        rect = Rect(left=left/self.target_width_mm, top=top/self.target_height_mm,
                    width=(self.marker_size_mm + 2*margin_mm)/self.target_width_mm,
                    height=(self.marker_size_mm + 2*margin_mm)/self.target_height_mm)
        offset = Point(x=margin_mm/(self.marker_size_mm + 2*margin_mm),
                       y=margin_mm/(self.marker_size_mm + 2*margin_mm))
        return rect, offset
    
    def _compute_reference_grid(self):
        #TODO doc
        debug = True
        num_refpts_per_row = self.circles_per_row - 1
        num_refpts_per_col = self.circles_per_col - 1
        
        
        if debug:
            from vito import imvis, imutils
            import cv2
            vis = imutils.ensure_c3(self.calibration_template.tpl_full.copy())

        visited = np.zeros((num_refpts_per_col, num_refpts_per_row), dtype=np.bool)
        nodes_to_visit = deque()
        nodes_to_visit.append(self._make_reference_point(0, 0))
        nnr = 0
        reference_points = list()
        while nodes_to_visit:
            n = nodes_to_visit.popleft()
            vidx = self._refpt2posgrid(n.col, n.row)
            if vidx is None or visited[vidx.row, vidx.col]:
                continue
            nnr += 1
            visited[vidx.row, vidx.col] = True
            ## 8-neighborhood
            nodes_to_visit.append(self._make_reference_point(col=n.col,   row=n.row-1))
            nodes_to_visit.append(self._make_reference_point(col=n.col-1, row=n.row-1))
            nodes_to_visit.append(self._make_reference_point(col=n.col-1, row=n.row))
            nodes_to_visit.append(self._make_reference_point(col=n.col-1, row=n.row+1))
            nodes_to_visit.append(self._make_reference_point(col=n.col,   row=n.row+1))
            nodes_to_visit.append(self._make_reference_point(col=n.col+1, row=n.row+1))
            nodes_to_visit.append(self._make_reference_point(col=n.col+1, row=n.row))
            nodes_to_visit.append(self._make_reference_point(col=n.col+1, row=n.row-1))
            ## 4-neighborhood
            # nodes_to_visit.append(self._make_reference_point(col=n.col,   row=n.row-1))
            # nodes_to_visit.append(self._make_reference_point(col=n.col-1, row=n.row))
            # nodes_to_visit.append(self._make_reference_point(col=n.col,   row=n.row+1))
            # nodes_to_visit.append(self._make_reference_point(col=n.col+1, row=n.row))

            if n.surrounding_circles is not None:
                reference_points.append(n)
                if debug:
                    pt = self._mm2px(n.pos_mm_tl)
                    cv2.putText(vis, f'{nnr:d}', pt.int_repr(), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    cv2.circle(vis, pt.int_repr(), 3, (255, 0, 0), -1)
                    imvis.imshow(vis, "Reference Grid", wait_ms=1)
        object.__setattr__(self, 'reference_points', reference_points)
        if debug:
            print('Check "reference grid". Press key to continue.')
            imvis.imshow(vis, "Reference Grid", wait_ms=-1)
    
    def _make_reference_point(self, col, row):
        mm_tl = self._refpt2mm(col, row)
        mm_center = Point(x=col * self.dist_circles_mm, y=row * self.dist_circles_mm)
        circ = self._refpt2surrounding_circles(col, row)
        return ReferencePoint(col=col, row=row, pos_mm_tl=mm_tl, pos_mm_centered=mm_center, surrounding_circles=circ)

    def _mm2px(self, pt):
        return Point(x=pt.x / self.target_width_mm * self.calibration_template.tpl_full.shape[1],
                     y=pt.y / self.target_height_mm * self.calibration_template.tpl_full.shape[0])

    def _refpt2circlef(self, refgrid_col, refgrid_row):
        # Float circle "index" of square marker's center
        sqcenter_circ_x = self.square_topleft_corner.col + (self.circles_per_square_edge_length - 1)/2
        sqcenter_circ_y = self.square_topleft_corner.row + (self.circles_per_square_edge_length - 1)/2
        return (refgrid_col + sqcenter_circ_x, refgrid_row + sqcenter_circ_y)
    
    def _refpt2posgrid(self, refgrid_col, refgrid_row):
        # positive grid index (to be used for marking off visited refpoints)
        c, r = self._refpt2circlef(refgrid_col, refgrid_row)
        # print('* ', refpt, c, r)
        c = int(np.floor(c))
        r = int(np.floor(r))
        # print('=> ', c, r)
        if c < 0 or c >= self.circles_per_row - 1:
            return None
        if r < 0 or r >= self.circles_per_col - 1:
            return None
        return GridIndex(col=c, row=r)
    
    def _refpt2surrounding_circles(self, refgrid_col, refgrid_row):
        # Float circle "index" of square marker's center
        c, r = self._refpt2circlef(refgrid_col, refgrid_row)
        # print('* ', refpt, c, r)
        left = int(np.floor(c))
        top = int(np.floor(r))
        if left < 0 or left >= self.circles_per_row:
            return None
        if top < 0 or top >= self.circles_per_col:
            return None
        indices = [GridIndex(row=top, col=left),
                   GridIndex(row=top+1, col=left),
                   GridIndex(row=top+1, col=left+1),
                   GridIndex(row=top, col=left+1)]
        if any([self._skip_circle_idx(idx.row, idx.col) for idx in indices]):
            return None
        return indices

    def _refpt2mm(self, refgrid_col, refgrid_row):
        # Get (float) indices
        col, row = self._refpt2circlef(refgrid_col, refgrid_row)
        mx, my = self.computed_margins
        mm_first_circ_offset_x = mx + self.r_circles_mm
        mm_first_circ_offset_y = my + self.r_circles_mm
        return Point(x=mm_first_circ_offset_x + col * self.dist_circles_mm,
                     y=mm_first_circ_offset_y + row * self.dist_circles_mm)

    
    def _compute_calibration_template(self):
        """Precomputes the calibration template image and reference points."""
        # Render full template
        tpl_full = imutils.grayscale(self.render_image())  # The rendered PNG will have 3-channels
        tpl_h, tpl_w = tpl_full.shape[:2]

        # Compute center marker position relative to the full target
        relrect_marker_tight, _ = self._get_relative_marker_rect(0)
        marker_roi_tight = Rect(
                left=tpl_w * relrect_marker_tight.left,
                top=tpl_h * relrect_marker_tight.top,
                width=tpl_w * relrect_marker_tight.width,
                height=tpl_h * relrect_marker_tight.height)
        # Compute the corresponding corners for homography estimation (full
        # image warping) and ensure that they are sorted in CCW order
        ref_pts_marker_absolute = sort_points_ccw([
                marker_roi_tight.top_left,
                marker_roi_tight.bottom_left,
                marker_roi_tight.bottom_right,
                marker_roi_tight.top_right])

        # Crop the central marker (with some small margin) and resize
        # to the configured template size
        relrect_marker_padded, rmr_offset = \
            self._get_relative_marker_rect(self.template_marker_margin_mm)
        marker_roi_padded = Rect(
                left=tpl_w * relrect_marker_padded.left,
                top=tpl_h * relrect_marker_padded.top,
                width=tpl_w * relrect_marker_padded.width,
                height=tpl_h * relrect_marker_padded.height)
        tpl_marker = cv2.resize(imutils.crop(tpl_full, marker_roi_padded.int_repr()),
                                dsize=(self.template_marker_size_px,
                                       self.template_marker_size_px),
                                interpolation=cv2.INTER_CUBIC)
        # Compute the reference corners for homography estimation and ensure that
        # they are sorted in CCW order
        ref_pts_tpl_marker = sort_points_ccw([
                Point(x=rmr_offset.x * tpl_marker.shape[1],
                      y=rmr_offset.y * tpl_marker.shape[0]),
                Point(x=tpl_marker.shape[1] - rmr_offset.x * tpl_marker.shape[1],
                      y=rmr_offset.y * tpl_marker.shape[0]),
                Point(x=tpl_marker.shape[1] - rmr_offset.x * tpl_marker.shape[1],
                      y=tpl_marker.shape[0] - rmr_offset.y * tpl_marker.shape[0]),
                Point(x=rmr_offset.x * tpl_marker.shape[1],
                      y=tpl_marker.shape[0] - rmr_offset.y * tpl_marker.shape[0])])

        # Compute the circle template to locate the actual calibration reference
        # points (i.e. the grid points)
        relrect_circle, ref_pts_circle_relative = self._get_relative_marker_circle_v2()
        circle_roi = Rect(
            left=tpl_w * relrect_circle.left,
            top=tpl_h * relrect_circle.top,
            width=tpl_w * relrect_circle.width,
            height=tpl_h * relrect_circle.height)
        # Ensure the ROI is even (simplifies using the correlation results later on)
        circle_roi.ensure_odd_size()
        tpl_circle = imutils.crop(tpl_full, circle_roi.int_repr())
        ref_pts_tpl_circle = [Point(x=pt.x * tpl_circle.shape[1],
                                    y=pt.y * tpl_circle.shape[0]) for pt in ref_pts_circle_relative]
        # Compute expected size of a circle in the image template
        dia_circle_px = self.dia_circles_mm / self.target_width_mm * tpl_full.shape[1]
        ###DEBUG VISUALS
        # vis_tpl_circle = tpl_circle.copy()
        # foo = imutils.ensure_c3(tpl_full.copy()) # FIXME REMOVE
        # cv2.rectangle(foo, circle_roi.int_repr(), color=(0, 250, 0), thickness=3)
        # for pt in ref_pts_tpl_circle:
        #     cv2.circle(vis_tpl_circle, tuple(pt.int_repr()), radius=3, color=(200, 0, 0), thickness=1)
        # imvis.imshow(foo, 'FOO', wait_ms=10)
        # print(tpl_circle.shape, ref_pts_tpl_circle[0])
        # imvis.imshow(vis_tpl_circle, "CIRCLE????", wait_ms=-1)

        object.__setattr__(self, 'calibration_template', CalibrationTemplate(
                                        tpl_full=tpl_full,
                                        refpts_full_marker=ref_pts_marker_absolute,
                                        #TODO extract reference points!!!! grid
                                        tpl_cropped_marker=tpl_marker,
                                        refpts_cropped_marker=ref_pts_tpl_marker,
                                        tpl_cropped_circle=tpl_circle,
                                        refpts_cropped_circle=ref_pts_tpl_circle,
                                        dia_circle_px=dia_circle_px
                                    ))


# # """Test pattern for development."""
# eddie_test_specs_a4 = PatternSpecificationEddie('Eddie Test Pattern A4',
#     target_width_mm=210, target_height_mm=297,
#     dia_circles_mm=5, dist_circles_mm=11)


def save_eddie_assets():
    """Export the pre-configured targets to the module's export folder."""
    folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                          'exported')
    export_pattern(eddie_test_specs_a4, folder,
                   None, export_pdf=True, export_png=True,
                   prevent_overwrite=False)

    
if __name__ == '__main__':
    from ..export import export_pattern
    import os
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    save_eddie_assets()
