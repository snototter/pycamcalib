from collections import namedtuple
from dataclasses import dataclass

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

#TODO sort points cw/ccw
#https://www.baeldung.com/cs/sort-points-clockwise
#https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
# https://github.com/snototter/vitocpp/blob/master/src/cpp/vcp_math/geometry2d.cpp#L1113