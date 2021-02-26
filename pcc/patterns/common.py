from collections import namedtuple
from dataclasses import dataclass
from functools import cmp_to_key

GridIndex = namedtuple('GridIndex', 'row col')


@dataclass
class Point:
    x: float
    y: float
    def to_int(self):
        return [int(x) for x in [self.x, self.y]]


_Rect = namedtuple('_Rect', 'left top width height')


class Rect(_Rect):
    def to_int(self):
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
    # Cross product of the vectors (p1 - pt_ref) and (p2 - pt_ref) yields
    # the signed area:
    area_rect = (pt1.x - pt_ref.x)*(pt2.y - pt_ref.y)\
                - (pt1.y - pt_ref.y)*(pt2.x - pt_ref.x)
    if area_rect == 0:
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
                  reverse=True)
