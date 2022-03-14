from . import (
    PICKLE_ROOT,
    PROJECT_ROOT,
    _doppelganger,
    _pickle_name,
    _validate_graph_name,
)

from .local_separators import (
    Vertex,
    LocalCutvertex,
    is_local_cutvertex,
    ball,
    find_local_cutvertices as flc,
)

from pathlib import Path
from typing import (
    Generator,
    List,
    Set,
    Tuple,
)

import pickle
import networkx as nx

def tmp_pickle_name(path: Path) -> Path:
    return _doppelganger(PICKLE_ROOT, path, '.incremental')

def flc_incremental(G: nx.Graph, min_locality: int=3) -> List[LocalCutvertex]:
    # To determine if a vertex is indeed a local separator and not a component-level
    # separator, we'll need the connected components in the graph.
    components: List[Set[Vertex]] = list(nx.connected_components(G))
    # Prepare the return value list.
    local_cutvertices: List[LocalCutvertex] = []
    # Iterate through each vertex in the graph.
    for v in G.nodes:
        # Prepare the binary search bounds.
        mi: int = min_locality
        component: Set[Vertex] = next(comp for comp in components if v in comp)
        ma: int = len(component)
        if mi > ma:
            # This can occur is there are fewer vertices in the
            # component v belongs to than expected. In this case,
            # the radius we would potentially determine for v is
            # not of interest, given that it would be smaller than
            # min_radius, hence we should move onto the next vertex.
            continue
        mid: int = ma
        while mid >= mi:
            if is_local_cutvertex(G, v, mid):
                component_without_v: nx.Graph = nx.classes.graphviews.subgraph_view(
                    G, filter_node=lambda x: x in component and x != v
                )
                # Run the component connectedness check.
                if nx.algorithms.components.is_connected(component_without_v):
                    # v does not separate its component, hence it's a genuine local separator.
                    # Before appending v to the list of local cutvertices we need to obtain the
                    # components of B_{r/2}(v)-v and track the edges they send to v in G.
                    # Step 1: Obtain the punctured ball of radius r/2 around v.
                    punctured_ball: nx.Graph = nx.classes.graphviews.subgraph_view(
                        ball(G, v, mid/2), filter_node=lambda x: x != v
                    )
                    # Step 2: Obtain the connected components of the punctured ball.
                    punctured_ball_components: Generator[Set[Vertex], None, None] = nx.connected_components(punctured_ball)
                    # Step 3: Obtain the neighbourhood of v in G.
                    neighbourhood: Set[Vertex] = set(G.neighbors(v))
                    # Step 4: Obtain the intersections of the neighbourhood of v in G with
                    #         the components of the punctured ball.
                    edge_partition: Set[Tuple[Vertex, ...]] = set(
                        tuple(neighbourhood.intersection(comp)) for comp in punctured_ball_components
                    )
                    # Add LocalCutvertex v to the list of local cutvertices.
                    local_cutvertices.append(
                        LocalCutvertex(vertex=v, locality=mid, edge_partition=edge_partition)
                    )
                    break
                else:
                    # v separates its component, so it isn't a genuine local cutvertex.
                    # Keep searching
                    mid -= 1
            else:
                mid -= 1
    # Done iterating over the vertices.
    return local_cutvertices

def pickle_local_separators(G: nx.Graph, overwrite: bool=False):
    ### Validate the graph's name attribute.
    _validate_graph_name(G.name)
    ### Obtain pickling path.
    graph_pickle: Path = tmp_pickle_name(G.name)
    ### Handle the overwrite parameter.
    if graph_pickle.exists() and graph_pickle.stat().st_size and not overwrite:
        return
    ### Compute local cutvertices.
    local_cutvertices: List[LocalCutvertex] = flc_incremental(G)
    ### Make sure parent folder exists.
    graph_pickle.parent.mkdir(parents=True, exist_ok=True)
    ### Serialise result.
    with open(graph_pickle, 'wb') as handle:
        pickle.dump(local_cutvertices, handle, protocol=pickle.HIGHEST_PROTOCOL)

def __pickle_Network_Neural_MJS20(overwrite: bool=False):
    if overwrite:
        print('OVERWRITING PICKLED (INCREMENTALLY FOUND) LOCAL CUTVERTICES!')
        print('Confirm? [y/n]')
        while True:
            print('>>> ', end='')
            ans: str = input()
            if ans.lower().startswith('y'):
                print('Overwrite confirmed.')
                break
            elif ans.lower().startswith('n'):
                print('Overwrite aborted!')
                exit()
    from .datasets import Network_Data_MJS20 as ND
    from .utils import seconds_to_string as sec2str
    import time
    # Run routine to pickle Network_Neural_MJS20 dataset.
    categories: List[str] = 'FoodWebs Genetic Language Metabolic Neural Social Trade'.split()
    datasets: List[List[nx.Graph]] = list(map(lambda x: ND[x], categories))
    total_start: float = time.perf_counter()
    for i, (category, dataset) in enumerate(zip(categories, datasets)):
        print(f'[{i+1}/{len(categories)}] {category}')
        G: nx.Graph
        for i, G in enumerate(dataset):
            print(f'\t({i+1}/{len(dataset)}) Pickling {G.name.name}...')
            start = time.perf_counter()
            pickle_local_separators(G, overwrite=overwrite)
            end = time.perf_counter()
            print('\tTook', sec2str(end-start, rounding=3))
    total_end: float = time.perf_counter()
    dur: float = total_end - total_start
    print('Total pickling time:', sec2str(dur, rounding=2))

def compare_pickles():
    from .datasets import Network_Data_MJS20 as ND
    print('Comparing pickled results...')
    problematic: List[Tuple[nx.Graph, List[LocalCutvertex], List[LocalCutvertex]]] = []
    categories: List[str] = 'FoodWebs Genetic Language Metabolic Neural Social Trade'.split()
    datasets: List[List[nx.Graph]] = list(map(lambda x: ND[x], categories))
    graphs: List[nx.Graph] = [graph for dataset in datasets for graph in dataset]
    for graph in graph:
        with open(_pickle_name(G.name), 'rb') as handle:
            flc_bin: List[LocalCutvertex] = pickle.load(handle)
        with open(tmp_pickle_name(G.name), 'rb') as handle:
            flc_inc: List[LocalCutvertex] = pickle.load(handle)
        if flc_bin != flc_inc:
            problematic.append(
                (graph, flc_bin, flc_inc)
            )
    if not problematic:
        print('whew...')
        cleanup_inc_pickles()
    else:
        # oh boy
        with open(PROJECT_ROOT / 'bruh.nani', 'wb') as handle:
            pickle.dump(problematic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def cleanup_inc_pickles():
    print('DELETING PICKLED (INCREMENTALLY FOUND) LOCAL CUTVERTICES!')
    print('Confirm? [y/n]')
    while True:
        print('>>> ', end='')
        ans: str = input()
        if ans.lower().startswith('y'):
            print('Deletion confirmed.')
            break
        elif ans.lower().startswith('n'):
            print('Deletion aborted!')
            exit()
    from .datasets import Network_Data_MJS20 as ND
    categories: List[str] = 'FoodWebs Genetic Language Metabolic Neural Social Trade'.split()
    datasets: List[List[nx.Graph]] = list(map(lambda x: ND[x], categories))
    graphs: List[nx.Graph] = [graph for dataset in datasets for graph in dataset]
    for graph in graphs:
        tmp_pickle_name(G.name).unlink()
    print('Deleted pickled (incrementally found) local cutvertices.')

if __name__ == '__main__':
    __pickle_Network_Neural_MJS20(overwrite=False) # not risking shit
    compare_pickles()
