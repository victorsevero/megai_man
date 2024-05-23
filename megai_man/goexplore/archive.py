from collections import defaultdict
from sys import getsizeof as size

from .cell import Cell


class Archive(defaultdict):
    """Archive object subclassed from :class:`~collections.defaultdict`

    Overloads the :func:`~__sizeof__` method to compute the size of the archive in memory.
    """

    def __init__(self):
        super().__init__(Cell)

    def __sizeof__(self):
        total = 0
        for code, cell in self.items():
            total += size(code) + size(cell)
        return total
