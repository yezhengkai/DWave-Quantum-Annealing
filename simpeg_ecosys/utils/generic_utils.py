"""For generic purpose use."""
from typing import Iterable
from itertools import groupby


def flatten(iterable):
    """Yield items from any nested iterable

    Parameters
    ----------
    iterable : Iterable exclude str and bytes
        Iterable object to be flattened.

    Yields
    ------
    generator
        To generate flatten iterable object.

    References
    ----------
    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    """
    for x in iterable:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def all_equal(iterable):
    """Check if all elements in the iterable object are identical

    Parameters
    ----------
    iterable : Iterable
        The iterable object to check.

    Returns
    -------
    bool
        True if all elements in the iterable object are the same

    References
    ----------
    https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
