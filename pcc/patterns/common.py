
from collections import namedtuple
GridIndex = namedtuple('GridIndex', 'row col')
_Point = namedtuple('Point', 'x y')
class Point(_Point):
    def to_int(self):
        return [int(x) for x in [self.x, self.y]]

_Rect = namedtuple('_Rect', 'left top width height')

class Rect(_Rect):
    def to_int(self):
        return [int(x) for x in [self.left, self.top, self.width, self.height]]

class SpecificationError(Exception):
    """Raised for invalid pattern specifications."""
    pass
