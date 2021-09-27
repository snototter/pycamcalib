import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from functools import cmp_to_key


GridIndex = namedtuple('GridIndex', 'row col')


@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float

    def int_repr(self):
        """Returns a tuple representation [x, y] with all coordinates truncated
        to integers (a tuple is required for OpenCV compatibility)."""
        return (int(self.x), int(self.y))

    def distance(self, pt):
        """Returns the distance to the given point."""
        return np.sqrt((pt.x - self.x)**2 + (pt.y - self.y)**2)

    def dot(self, pt):
        """Returns the dot product between this and the given 2d vector."""
        return self.x * pt.x + self.y * pt.y
    

@dataclass
class Edge:
    """Represents a 2d edge."""
    pt1: Point
    pt2: Point

    @property
    def length(self):
        """Returns the length of this edge (distance between the two endpoints)."""
        return self.pt1.distance(self.pt2)
    
    @property
    def direction(self):
        """Returns the unnormalized direction vector."""
        return Point(x=self.pt2.x - self.pt1.x, y=self.pt2.y - self.pt1.y)

    @property    
    def unit_direction(self):
        """Returns the unit direction vector."""
        dirvec = self.direction
        length = self.length
        if length > 0:
            return Point(x=dirvec.x / length, y=dirvec.y / length)
        return dirvec

    @property
    def homogenous_form(self):
        """Returns the homogenous representation of this line in projective P2 space."""
        # For more details on lines in projective space:
        # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
        # http://robotics.stanford.edu/~birch/projective/node4.html
        a = np.array([self.pt1.x, self.pt1.y, 1], dtype=np.float32)
        b = np.array([self.pt2.x, self.pt2.y, 1], dtype=np.float32)
        return np.cross(a, b)
    
    def angle(self, e2):
        """Returns the angle (in radians) between this and the given edge."""
        dp = self.unit_direction.dot(e2.unit_direction)
        return np.arccos(max(-1, min(1, dp)))

    def intersection(self, e2):
        """Return the intersection point (or None) between this and the given edge."""
        # In P2, line intersection is simply their cross product:
        ip = np.cross(self.homogenous_form, e2.homogenous_form)
        if np.isclose(ip[2], 0, atol=1e-7):
            # Parallel edges
            return None
        # Normalize (P2 space ==> R2, i.e. Euclidean 2-space)
        return Point(x=ip[0] / ip[2], y=ip[1] / ip[2])


@dataclass
class Rect:
    """Represents a rectangle."""
    left: float
    top: float
    width: float
    height: float

    @property
    def top_left(self):
        return Point(x=self.left, y=self.top)

    @property
    def top_right(self):
        return Point(x=self.left+self.width, y=self.top)
    
    @property
    def bottom_right(self):
        return Point(x=self.left+self.width, y=self.top+self.height)
    
    @property
    def bottom_left(self):
        return Point(x=self.left, y=self.top+self.height)

    def int_repr(self):
        """Returns a list representation [left, top, width, height] with all values truncated to integers."""
        return [int(x) for x in [self.left, self.top, self.width, self.height]]
    
    def ensure_odd_size(self):
        """Adjusts the width & height s.t. the width and height are odd."""
        if int(self.width) % 2 == 0:
            self.width += 1
        if int(self.height) % 2 == 0:
            self.height += 1


class SpecificationError(Exception):
    """Raised for invalid pattern specifications."""
    pass


def center(ptlist):
    """Returns the barycenter of the given list of Point instances."""
    N = len(ptlist)
    if N == 0:
        return None
    sum_ = Point(0, 0)
    for pt in ptlist:
        sum_.x += pt.x
        sum_.y += pt.y
    return Point(x=sum_.x/N, y=sum_.y/N)


def ccw_cmp(pt_ref, pt1, pt2):
    """Comparator to sort 2d points w.r.t. to a reference
    point in counter-clockwise order."""
    # See pseudocode from Wiki https://en.wikipedia.org/wiki/Graham_scan
    # Cross product of the vectors (p1 - pt_ref) and (p2 - pt_ref) yields
    # the signed area:
    area_rect = (pt1.x - pt_ref.x)*(pt2.y - pt_ref.y)\
                - (pt1.y - pt_ref.y)*(pt2.x - pt_ref.x)
    if area_rect == 0:
        # All three points are collinear. Thus, sort pt1 and pt2 by their distance to ref:
        #FIXME verify collinear case on paper!!!
        if pt_ref.distance(pt1) < pt_ref.distance(pt2):
            return 1
        else:
            return 0
        return 0
    elif area_rect > 0:
        return 1
    return -1


def sort_points_ccw(pt_list, pt_ref=None, reverse=False):
    """
    Sorts the given list of 3d coordinates counter-clockwise
    around the given reference point. If pt_ref is None, the barycenter
    will be used as reference point. To get a clockwise-sorted list,
    use reverse=True.
    """
    if pt_ref is None:
        pt_ref = center(pt_list)
    return sorted(pt_list,
                  key=cmp_to_key(lambda pt1, pt2: ccw_cmp(pt_ref, pt1, pt2)),
                  reverse=reverse)


def image_corners(x):
    if isinstance(x, np.ndarray):
        height, width = x.shape[:2]
    elif isinstance(x, list) or isinstance(x, tuple):
        width, height = x[:2]
    else:
        raise RuntimeError('Cannot interpret input (must be either np.ndarray, list, or tuple)')
    return [Point(0, 0), Point(width, 0), Point(width, height), Point(0, height)]


def numpy2cvpt(nppt):
    return (int(nppt[0]), int(nppt[1]))


def points2numpy(pt_list, Nx2=True):
    #TODO asert or return error code (maybe later)
    # opencv bindings require list of opints to be Nx2 dimensional ndarrays
    assert pt_list is not None
    assert len(pt_list) > 0
    if Nx2:
        npp = np.zeros((len(pt_list), 2), dtype=np.float32)
        for idx in range(len(pt_list)):
            npp[idx, 0] = pt_list[idx].x
            npp[idx, 1] = pt_list[idx].y
    else:
        npp = np.zeros((2, len(pt_list)), dtype=np.float32)
        for idx in range(len(pt_list)):
            npp[0, idx] = pt_list[idx].x
            npp[1, idx] = pt_list[idx].y
    return npp


def bottommost_point(pt_list):
    if pt_list is None or len(pt_list) == 0:
        return None
    bm_idx = 0
    for idx in range(len(pt_list)):
        if pt_list[idx].y < pt_list[bm_idx].y:
            bm_idx = idx
    return pt_list[bm_idx]


def fully_qualified_classname(obj):
    """
    Returns the fully qualified class name of the given object.
    Adapted from https://stackoverflow.com/a/2020083
    """
    c = obj.__class__
    m = c.__module__
    if m == 'builtins':  # Avoid returning "builtins.str"
        return c.__qualname__
    return m + '.' + c.__qualname__


def paper_format_str(width_mm, height_mm):
    """
    Returns the paper format corresponding to the given paper dimensions.
    If the size is not mapped to a format, a string representation 'Wmm x Hmm'
    will be returned instead.
    """
    formats = {
        (841, 1189): 'A0',  # German: "Vierfachbogen"
        (594, 841): 'A1',  # "Doppelbogen"
        (420, 594): 'A2',  # "Bogen"
        (297, 420): 'A3',  # "Halbbogen"
        (210, 297): 'A4',  # "Viertelbogen"
        (148, 210): 'A5'  # "Blatt/Achtelbogen"
    }
    key = (width_mm, height_mm)
    if key in formats:
        return formats[key]
    return f'{width_mm}mm x {height_mm}mm'
