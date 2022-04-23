from . import (
    DatasetInterface,
    DATASETS_ROOT,
)
from ..local_separators import Vertex
from ..utils import (
    collinear,
    path_to_str,
)

from pathlib import Path
from typing import (
    Iterator,
    List,
    Tuple,
    Set,
)

import networkx as nx
import pickle
import shapefile
import sys

SAVE_PROGRESS_EVERY: int = 100
DATASET_FOLDER: Path = DATASETS_ROOT / 'MajorOpenRoadNetworks'
OVERLAPPING_GROUPS_FILENAME: Path = DATASET_FOLDER / 'OverlappingGroups.pickle'
FLATTEN_GROUPS_FILENAME: Path = DATASET_FOLDER / 'FlattenGroups.pickle'
POST_PROCESSED_GRAPH_FILENAME: Path = DATASET_FOLDER / 'PostProcessedGraph.pickle'

def __get_overlapping_groups(G: nx.Graph) -> List[Set[Vertex]]:
    '''
        Function to construct groups consisting of vertices overlapping
        at the same latitude/longitude pair.

        Parameters
        ----------
        G: nx.Graph
            The graph to operate on, assumed to be MORN.
        
        Returns
        -------
        List[Set[Vertex]]
    '''
    # Obtain dictionary of latitude/longitude pairs.
    pos = G.graph['pos']
    # Check for checkpoint.
    if not (OVERLAPPING_GROUPS_FILENAME.exists() and OVERLAPPING_GROUPS_FILENAME.stat().st_size > 0):
        # No checkpoint, starting routine from scratch.
        checkpoint: int = 0
        groups: List[Set[Vertex]] = []
    else:
        # Checkpoint found, attempting loading data.
        checkpoint: int
        groups: List[Set[Vertex]]
        with open(OVERLAPPING_GROUPS_FILENAME, 'rb') as handle:
            data = pickle.load(handle)
            try:
                checkpoint, groups = data
            except ValueError:
                # Assume routine previously ran to completion and return result.
                groups = data
                return groups
    print(f'<MORN> Obtaining overlapping groups of vertices{f" (using checkpoint {checkpoint}/{G.number_of_nodes()})" * bool(checkpoint)}...')
    # Obtain ordered list of nodes.
    nodes: List[Vertex] = list(G.nodes())
    # Proceed to finding overlapping groups.
    for i, node in enumerate(nodes[checkpoint:]):
        # Save progress.
        if i and not i % SAVE_PROGRESS_EVERY:
            with open(OVERLAPPING_GROUPS_FILENAME, 'wb') as handle:
                pickle.dump((i, groups), handle)
        # Begin by checking if a group has already been created for either the group
        # coordinator or its closest vertex.
        try:
            group: Set[Vertex] = next(group for group in groups if node in group)
            # Group found, carry on.
            continue
        except StopIteration:
            # A group entry hasn't been made for this node, proceed with the routine.
            pass
        # Create starting group.
        group: Set[Vertex] = {node}
        # Go through vertices hoping to expand the group.
        for other_node in nodes:
            if pos[node] == pos[other_node]:
                group.add(other_node)
        # Add group to list of groups.
        groups.append(group)
    # Routine complete, save total progres without checkpoint.
    with open(OVERLAPPING_GROUPS_FILENAME, 'wb') as handle:
        pickle.dump(groups, handle)
    # Return overlapping groups.
    return groups

def __flatten_overlapping_groups(G: nx.Graph, groups: List[Set[Vertex]]) -> Tuple[List[int], nx.Graph]:
    '''
        Takes a graph and a list of groups of nodes whose latitude/longitude
        pairs overlap, and flattens the graph along those groups.

        Parameters
        ----------
        G: nx.Graph
            The graph to operate on, assumed to be MORN.
        groups: List[Set[Vertex]]
            The list of groups with overlapping latitude/longitude pairs.
        
        Notes
        -----
        This function keeps track of the number of components in the graph
        while the operation is being carried out for statistics purposes.

        Returns
        -------
        Tuple[List[int], nx.Graph]
            List[int]
                The number of components in the graph as the operation was
                performed.
            nx.Graph
                The resulting graph.
    '''
    # Obtain dictionary of latitude/longitude pairs.
    pos = G.graph['pos']
    # Check for checkpoint.
    if not (FLATTEN_GROUPS_FILENAME.exists() and FLATTEN_GROUPS_FILENAME.stat().st_size > 0):
        # No checkpoint, start from scratch.
        checkpoint: int = 0
        component_count_accumulator: List[int] = [nx.number_connected_components(G)]
    else:
        # Checkpoint found, attempting loading data.
        checkpoint: int
        component_count_accumulator: List[int]
        with open(FLATTEN_GROUPS_FILENAME, 'rb') as handle:
            data = pickle.load(handle)
        try:
            checkpoint, component_count_accumulator, G = data
        except ValueError:
            # Assume routine previously ran to completion and return result.
            component_count_accumulator, G = data
            return component_count_accumulator, G
    
    def __flatten_group(group: Set[Vertex]):
        assert len(group) > 1
        group_iter: Iterator[Vertex] = iter(group)
        vertex: Vertex = next(group_iter)
        to_delete: List[Vertex] = list(group_iter)
        # Create the merged neighbourhood.
        merged_neighbourhood: Set[Vertex] = set()
        merged_neighbourhood.update(*(set(G.neighbors(v)) for v in to_delete))
        # Add edges from vertex to each neighbour in the merged neighbourhood.
        G.add_edges_from((vertex, v) for v in merged_neighbourhood)
        # Remove the nodes marked for deletion.
        G.remove_nodes_from(to_delete)
        # Remove them from the dictionary of latitude/longitude pairs.
        for node in to_delete:
            del pos[node]

    print(f'<MORN> Flattening overlapping groups{f" (using checkpoint {checkpoint}/{len(groups)})" * bool(checkpoint)}...')
    # Process groups.
    for i, group in enumerate(groups[checkpoint:]):
        # Save progress.
        if i and not i % SAVE_PROGRESS_EVERY:
            with open(FLATTEN_GROUPS_FILENAME, 'wb') as handle:
                pickle.dump((i, component_count_accumulator, G), handle)
        # Flatten group and obtain component count.
        __flatten_group(group)
        component_count_accumulator.append(nx.number_connected_components(G))
    
    # Routine complete, save overall progress.
    with open(FLATTEN_GROUPS_FILENAME, 'wb') as handle:
        pickle.dump((component_count_accumulator, G), handle)
    
    # Return result.
    return component_count_accumulator, G

def __get_redundant_vertices(G: nx.Graph) -> List[Vertex]:
    '''
        Function to return vertices which are redundant.

        Parameters
        ----------
        G: nx.Graph
            The graph to operate on, assumed to be MORN.
        
        Notes
        -----
        A vertex is redundant if it has degree 2 and is collinear with
        both of its neighbours.

        Returns
        -------
        List[Vertex]
    '''
    print('<MORN> Identifying redundant vertices...')
    redundant: List[Vertex] = []
    pos = G.graph['pos']
    for vertex, X in pos.items():
        if G.degree(vertex) != 2:
            continue
        Y, Z = [pos[x] for x in G.neighbors(vertex)]
        if collinear(X, Y, Z):
            redundant.append(vertex)
    return redundant

def __remove_redundant_vertices(G: nx.Graph, redundant: List[Vertex]):
    '''
        Removes a list of assumed to be redundant vertices from a graph.

        Parameters
        ----------
        G: nx.Graph
            The graph to operate on, assumed to be MORN.
        redundant: List[Vertex]
            The redundant vertices.
        
        Notes
        -----
        This operation is performed in-place.
    '''
    ## NOTE: recall a redundant vertex v has 2 neighbours, hence
    ##       tuple(G.neighbors(v)) creates an edge between its neighbours.
    # Create edges around the redundant vertices.
    G.add_edges_from([tuple(G.neighbors(v)) for v in redundant])
    # Delete redundant vertices.
    G.remove_nodes_from(redundant)
    # Remove redundant vertices from the dictionary of latitude/longitude pairs.
    for node in redundant:
        del G.graph['pos'][node]
    
def parse_MajorOpenRoadNetworks_dataset(file: Path) -> nx.Graph:
    '''
        Parses the Major_Road_Network_2018_Open_Roads.zip dataset.
    '''
    # Check if the graph has already been post-processed.
    if POST_PROCESSED_GRAPH_FILENAME.exists() and POST_PROCESSED_GRAPH_FILENAME.stat().st_size > 0:
        # Graph has been previously fully post-processed, read it and return it.
        with open(POST_PROCESSED_GRAPH_FILENAME, 'rb') as handle:
            G: nx.Graph = pickle.load(handle)
        return G
    
    # Graph hasn't been post-processed, go through regular routine.
    # Create the empty graph.
    G: nx.Graph = nx.Graph(name=Path(file), pos=dict())
    # Read in the shapefile.
    with shapefile.Reader(path_to_str(file)) as shp:
        # Add polyline points as nodes.
        index: int = 0
        # print('Reading in points...')
        for shprec in shp.shapeRecords():
            offset: int = len(shprec.shape.points)
            G.add_nodes_from(range(index, index + offset))
            G.graph['pos'].update({
                index + i: point
                for i, point in enumerate(shprec.shape.points)
            })
            G.add_edges_from(
                (pre, nex)
                for pre, nex in zip(
                    range(index, index + offset - 1),
                    range(index + 1, index + offset)
                )
            )
            index += offset
    # Initial graph has been read from the shapefile, onto some pre-processing.
    
    ## Obtain overlapping groups.
    groups: List[Set[Vertex]] = __get_overlapping_groups(G)
    ## Flatten overlapping groups.
    component_count_accumulator, G = __flatten_overlapping_groups(G, groups)
    G.graph['component_count_accumulator'] = component_count_accumulator
    ## Obtain redundant vertices.
    redundant: List[Vertex] = __get_redundant_vertices(G)
    num_nodes: int = G.number_of_nodes()
    savings_ratio: float = 1 - (num_nodes - len(redundant)) / num_nodes
    G.graph['savings_ratio'] = savings_ratio
    ## Remove redundant vertices.
    __remove_redundant_vertices(G, redundant)

    # Save the post-processed graph.
    with open(POST_PROCESSED_GRAPH_FILENAME, 'wb') as handle:
        pickle.dump(G, handle)
    
    # Return the graph.
    return G

_DATASET_EXTENSIONS: Set[str] = {'.zip'}
_PROBLEMATIC_DATASETS: Tuple[str] = ()
MajorOpenRoadNetworks: DatasetInterface = DatasetInterface(
    __file__, _DATASET_EXTENSIONS, _PROBLEMATIC_DATASETS, parse_MajorOpenRoadNetworks_dataset
)

sys.modules[__name__] = MajorOpenRoadNetworks
