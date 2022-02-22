'''
    Utility functions used here and there during debugging.
'''

from .local_separators import Vertex

from math import sqrt
from typing import (
    Dict,
    List,
    Tuple,
    TypeVar,
    Union,
)

import inflect
import networkx as nx
import numpy as np
import numpy.typing as npt

Point2d = npt.NDArray[np.float64]
FigureSize = Tuple[float, float]

A4: FigureSize = (8.25, 11.75)

class BoundingBox2d:
    STR_ROUNDING: int       = 3
    STR_TEMPLATE: str       = 'BoundingBox(\n{}\n)'
    STR_FIELDS: List[str]   = 'top_left top_right bottom_left bottom_right'.split()
    def __init__(self, top_left, top_right, bottom_left, bottom_right):
        self.data = np.zeros((4, 2))
        self.data[0, :] = top_left
        self.data[1, :] = top_right
        self.data[2, :] = bottom_left
        self.data[3, :] = bottom_right
    
    def __str__(self) -> str:
        return self.STR_TEMPLATE.format(
            '\n'.join(
                f'\t{x}={tuple(self.data[i, :].round(self.STR_ROUNDING))}'
                for i, x in enumerate(self.STR_FIELDS)
            )
        )
    
    def __repr__(self) -> str:
        return str(self)

    @property
    def top_left(self) -> Point2d:
        return self.data[0, :]
    @top_left.setter
    def top_left(self, v):
        self.data[0, :] = v
    @property
    def top_right(self) -> Point2d:
        return self.data[1, :]
    @top_right.setter
    def top_right(self, v):
        self.data[1, :] = v
    @property
    def bottom_left(self) -> Point2d:
        return self.data[2, :]
    @bottom_left.setter
    def bottom_left(self, v):
        self.data[2, :] = v
    @property
    def bottom_right(self) -> Point2d:
        return self.data[3, :]
    @bottom_right.setter
    def bottom_right(self, v):
        self.data[3, :] = v

_p = inflect.engine()
_p.classical(all=True) # ew @ vertexes

def pluralise(q: Union[int, float], word: str) -> str:
    return f'{q} {_p.plural(word, q)}'

def seconds_to_string(dur: float, rounding: int=3) -> str:
    '''
        Takes a duration measured in seconds and returns a string
        according to my liking.

        Parameters
        ----------
        dur: float
        
        Returns
        -------
        str
    '''
    hours: int = int(dur // 3600)
    minutes: int = int(dur // 60) % 60
    seconds: float = round(dur % 60, rounding)
    if not hours:
        if not minutes:
            return pluralise(round(dur, rounding), 'second')
        return ', '.join([pluralise(minutes, 'minute'), pluralise(seconds, 'second')])
    return ', '.join([pluralise(hours, 'hour'), pluralise(minutes, 'minute'), pluralise(seconds, 'second')])

def beer_me_a_graph(n: int=300, s: float=10, v: float=4, p_in: float=.5, p_out: float=.5, seed: int=0xdead_beef) -> nx.Graph:
    '''
        Creates a Gaussian random partition graph.

        Parameters
        ----------
        n: float, default 300
            Number of nodes in the graph
        s: float, default 10
            Mean cluster size
        v: float, default 4
            Shape parameter. The variance of cluster size distribution is s/v
        p_in: float, default 0.5
            Probability of intra cluster connection
        p_out: float, default 0.5
            Probability of inter cluster connection
        seed: int, default 0xdead_beef
    '''
    args = n, s, v, p_in, p_out
    kwargs = {'directed': False, 'seed': seed}
    return nx.gaussian_random_partition_graph(*args, **kwargs)

def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    '''
        Returns the Euclidean distance between tuples of equal length.

        Parameters
        ----------
        a: Tuple[float, ...]
        b: Tuple[float, ...]

        Returns
        -------
        float
    '''
    n: int = len(a)
    if not (n and n == len(b)):
        raise ValueError(f'cannot compute Euclidean distance between tuples of lengths {n} and {len(b)}')
    return sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def nearest(pos: Dict[Vertex, Point2d]) -> Dict[Vertex, Vertex]:
    '''
        Returns a dictionary that maps each Vertex to its nearest neighbour.

        Parameters
        ----------
        pos: Dict[Vertex, Point2d]
            A mapping of vertices to their relative positions on the plot.
        
        Notes
        -----
        This does not consider the trivial, reflexive approach.

        Returns
        -------
        Dict[Vertex, Vertex]
            A dictionary mapping every vertex to its nearest vertex.
    '''
    return {
        k: min(
            pos,
            key=lambda x: euclidean_distance(v, pos[x]) if x != k else float('inf')
        ) for k,v in pos.items()
    }

def bounding_box_2d(coordinates: List[Point2d], fudge: float=None) -> BoundingBox2d:
    '''
        Returns the size of a bounding box around supplied coordinates.

        Parameters
        ----------
        coordinates: List[Point2d]
            A list of Cartesian coordinates

        Notes
        -----
        The bounding box is exact.

        Returns
        -------
        BoundingBox2d
    '''
    if not coordinates:
        raise ValueError('cannot construct a bounding box around an empty sequence')
    coordinates: npt.NDArray[Point2d] = np.vstack(coordinates)
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)
    if fudge is not None:
        height: float = max_y - min_y
        width: float = max_x - min_x
        horizontal_fudge: float = fudge * width
        vertical_fudge: float = fudge * height
        min_x -= horizontal_fudge
        max_x += horizontal_fudge
        min_y -= vertical_fudge
        max_y += vertical_fudge
    top_left       = (min_x, max_y)
    top_right      = (max_x, max_y)
    bottom_left    = (min_x, min_y)
    bottom_right   = (max_x, min_y)
    return BoundingBox2d(top_left, top_right, bottom_left, bottom_right)

def escape_underscore(s: str) -> str:
    return s.replace('_', '\_')
