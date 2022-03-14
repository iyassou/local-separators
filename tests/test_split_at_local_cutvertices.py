from src.local_separators import (
    split_at_local_cutvertices,
    Vertex,
    LocalCutvertex,
)
from tests.utils import polygon

from itertools import combinations
from math import sqrt
from typing import (
    Generator,
    List,
    Tuple,
    Union
)

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import networkx as nx
import unittest

class TestSplitAtLocalCutvertices(unittest.TestCase):

    def test_w13_with_six_spokes_removed(self):
        '''
            Tests the split_at_local_cutvertices function on the wheel on 13 vertices
            with every other pair of consecutive spokes removed.
        '''
        # Construct G.
        hub: Vertex = '$v$'
        rim: nx.Graph = nx.cycle_graph(range(1, 13))
        spokes: List[Tuple[Vertex, Vertex]] = []
        for i in range(1, 13, 4):
            spokes.extend([(hub, i), (hub, i+1)])
        G: nx.Graph = rim.copy()
        G.add_edges_from(spokes)
        # Split at the hub with radius 3.
        radius: int = 3
        hub_cutvertex: LocalCutvertex = LocalCutvertex(
            vertex=hub, locality=radius, edge_partition={(1, 2), (5, 6), (9, 10)}
        )
        H: nx.Graph = split_at_local_cutvertices(G, [hub_cutvertex], inplace=False)
        # Set the drawing layout.
        ### NOTE:   this is slightly complicated because I want the graphs
        ###         to be drawn in a specific way.
        #### Obtain the drawing layout of the rim.
        pos_rim: Dict[Vertex, Tuple[float, float]] = nx.circular_layout(rim)
        #### Obtain the rim's centroid.
        rim_centroid: Tuple[float, float] = tuple(map(sum, zip(*pos_rim.values())))
        #### Construct G's drawing layout.
        pos_G: Dict[Vertex, Tuple[float, float]] = pos_rim.copy()
        pos_G[hub] = rim_centroid
        #### Obtain the drawing layout of the triangle of split vertices.
        ##### Decide the height of the triangle. I'll take a fraction of
        ##### the smallest point-to-point distance.
        smallest_distance: Tuple[float, float] = min(
            sqrt((a-x)**2 + (b-y)**2)
            for (a,b),(x,y) in combinations(pos_G.values(), 2)
        )
        height: float = 0.8 * smallest_distance
        triangle: Generator[Tuple[float, float], None, None] = polygon(3, height)
        #### Construct H's drawing layout.
        pos_H: Dict[Vertex, Tuple[float, float]] = pos_G.copy()
        pos_H.update({v: next(triangle) for v, split in H.nodes(data='split') if split})
        # Show the results for now.
        ax1 = plt.subplot(121)
        nx.draw(G, pos=pos_G, with_labels=True)
        ax1.set_title(f'Original graph $G$, the wheel graph $W_{{{12}}}$ with every other pair of spokes removed')
        ax2 = plt.subplot(122)
        nx.draw(H, pos=pos_H, with_labels=True)
        ax2.set_title(f'$G$ split at $v$ with radius {radius}')
        plt.show()
        self.assertTrue(input('Did it work? [y/n] ').lower().startswith('y'))
