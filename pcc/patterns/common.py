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
        """Returns a list representation [x, y] with all coordinates truncated to integers."""
        return [int(x) for x in [self.x, self.y]]

    def distance(self, pt):
        """Returns the distance to the given point."""
        return np.sqrt((pt.x - self.x)**2 + (pt.y - self.y)**2)


_Rect = namedtuple('_Rect', 'left top width height')


class Rect(_Rect):
    """Represents a rectangle."""
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


def points2numpy(pt_list):
    #TODO asert or return error code (maybe later)
    assert pt_list is not None
    assert len(pt_list) > 0
    npp = np.zeros((len(pt_list), 2), dtype=np.float32)
    for idx in range(len(pt_list)):
        npp[idx, 0] = pt_list[idx].x
        npp[idx, 1] = pt_list[idx].y
    return npp