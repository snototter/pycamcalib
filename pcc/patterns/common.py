
from collections import namedtuple
GridIndex = namedtuple('GridIndex', 'row col')
Rect = namedtuple('Rect', 'left top width height')

class SpecificationError(Exception):
    """Raised for invalid pattern specifications."""
    pass
