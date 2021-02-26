
from collections import namedtuple
GridIndex = namedtuple('GridIndex', 'row col')
_Rect = namedtuple('_Rect', 'left top width height')

class Rect(_Rect):
    def to_int(self):
        return [int(x) for x in [self.left, self.top, self.width, self.height]]

class SpecificationError(Exception):
    """Raised for invalid pattern specifications."""
    pass
