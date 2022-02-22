from math import (
    cos,
    pi,
    sin,
)
from typing import (
    Generator,
    Tuple,
)

def polygon(N: int, r: float, centroid: Tuple[float, float]=None) -> Generator:
    '''
        Generates the Cartesian coordinates of an N-sided polygon.

        Parameters
        ----------
        N: int
        r: float
            The polygon's radius
        centroid: Tuple[float, float], optional
            Centroid to offset the coordinates by
        
        Returns
        -------
        Generator[Tuple[float, float], None, None]
            An N-sided polygon's Cartesian coordinates generator
    '''
    def point(i: int) -> Tuple[float, float]:
        theta: float = 2 * pi * i / N
        return r * cos(theta), r * sin(theta)
    if centroid is not None:
        return (tuple(map(sum, zip(centroid, point(n)))) for n in range(N))
    return (point(n) for n in range(N))
