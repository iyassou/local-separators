from . import (
    DatasetInterface,
    _edge_from_line,
)
from ..utils import pluralise

from pathlib import Path
from typing import (
    List,
    Set,
    Tuple,
)

import networkx as nx
import sys

def parse_infpower_dataset(file: Path) -> nx.Graph:
    '''
        Parses the inf-power.mtx dataset.
    '''
    with open(file, 'r') as f:
        # Ignore the first 2 lines.
        _, _ = f.readline(), f.readline()
        # Third line contains the number of vertices,
        # twice, and the number of edges.
        _, num_vertices, num_edges = map(int, f.readline().strip().split())
        # Subsequent lines are edges.
        edges: List[Tuple[int, int]] = list(
            map(
                _edge_from_line,
                f.readlines()
            )
        )
    # Sanity check the number of edges.
    diff: int = num_edges - len(edges)
    if diff < 0:
        raise ValueError(f'missing {pluralise(diff, "edge")}')
    elif diff > 0:
        raise ValueError(f'surplus of {pluralise(diff, "edge")}')
    # Create the graph.
    G: nx.Graph = nx.Graph(
        filter(lambda x: x is not None, edges), # filtering None from the edgelist
        name=Path(file)
    )
    # Sanity check the number of vertices.
    diff: int = num_vertices - len(G.nodes)
    if diff < 0:
        raise ValueError(f'missing {pluralise(diff, "vertex")}')
    elif diff > 0:
        raise ValueError(f'surplus of {pluralise(diff, "vertex")}')
    # Looking good
    return G

_DATASET_EXTENSIONS: Set[str] = {'.mtx'}
_PROBLEMATIC_DATASETS: Tuple[str] = ()
infpower: DatasetInterface = DatasetInterface(
    __file__, _DATASET_EXTENSIONS, _PROBLEMATIC_DATASETS, parse_infpower_dataset
)

sys.modules[__name__] = infpower

