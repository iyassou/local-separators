from . import DatasetInterface

from pathlib import Path
from typing import (
    List,
    Tuple,
    Set,
)

import networkx as nx
import sys
import zipfile

def parse_stackoverflow_dataset(file: Path) -> nx.Graph:
    '''
        Parses the stackoverflow.zip dataset.
    '''
    # Create the empty graph.
    G: nx.Graph = nx.Graph(name=file)
    # Go through the .ZIP file and populate the graph.
    # We'll keep track of the groups as a graph-level
    # attribute to ease rendering later.
    groups: Set[int] = set()
    with zipfile.ZipFile(file) as myzip:
        # First the nodes and their sizes.
        with myzip.open('stack_network_nodes.csv') as myfile:
            # Discard the header line...
            _ = myfile.readline()
            # ... and keep reading until EOF.
            line: str = myfile.readline().strip()
            while line:
                # Parse line.
                node, group, nodesize = line.split(b',')
                group: int = int(group)
                nodesize: float = float(nodesize) * 10.
                # Track group.
                groups.add(group)
                # Add node to graph.
                G.add_node(node, group=group, nodesize=nodesize)
                # Read next line.
                line: str = myfile.readline().strip()
        # Second the edges and their weights.
        with myzip.open('stack_network_links.csv') as myfile:
            # Discard the header line...
            _ = myfile.readline()
            # ... and keep reading until EOF.
            line: str = myfile.readline().strip()
            edges: List[Tuple[str, str, float]] = []
            while line:
                # Parse line.
                source, target, weight = line.split(b',')
                weight: float = float(weight)
                # Add edge to list of edges.
                edges.append((source, target, weight))
                # Read next line.
                line: str = myfile.readline().strip()
            # Add all edges to the graph.
            G.add_weighted_edges_from(edges)
    # Add groups as graph attribute.
    G.graph['groups']: Set[int] = groups
    # Return the graph (lol)
    return G

_DATASET_EXTENSIONS: Set[str] = {'.zip'}
stackoverflow: DatasetInterface = DatasetInterface(
    __file__, _DATASET_EXTENSIONS, parse_stackoverflow_dataset
)

sys.modules[__name__] = stackoverflow
