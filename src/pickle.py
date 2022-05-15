'''
    pickle.py   -   
'''

from . import (
    _pickle_name,
    _validate_graph_name,
)
from .datasets import (
    DatasetInterface,
    GETITEM_RETURN_TYPE,
    Providence,
)
from .local_separators import (
    find_local_cutvertices,
    Vertex,
    LocalCutvertex,
)

from pathlib import Path
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

import networkx as nx
import pickle

def pickle_local_separators(G: nx.Graph, overwrite: bool=False):
    '''
        Computes and then pickles the local separators for a graph.

        Parameters
        ----------
        graph: nx.Graph
            The graph to pickle.
        overwrite: bool, default False
            Whether or not to overwrite a pre-existing pickle.

        Raises
        ------
        ValueError
            If the graph's name is neither a Path nor a str.
    '''
    ### Validate the graph's name attribute.
    _validate_graph_name(G.name)
    ### Obtain pickling path.
    graph_pickle: Path = _pickle_name(G.name)
    ### Handle the overwrite parameter.
    if graph_pickle.exists() and graph_pickle.stat().st_size and not overwrite:
        return
    ### Compute local cutvertices.
    local_cutvertices: List[LocalCutvertex] = find_local_cutvertices(G)
    ### Make sure parent folder exists.
    graph_pickle.parent.mkdir(parents=True, exist_ok=True)
    ### Serialise result.
    with open(graph_pickle, 'wb') as handle:
        pickle.dump(local_cutvertices, handle, protocol=pickle.HIGHEST_PROTOCOL)

def _overwrite_guard(overwrite: bool):
    if overwrite:
        print('OVERWRITING PICKLED LOCAL CUTVERTICES!')
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

def __pickle_Network_Data_MJS20(overwrite: bool=False):
    _overwrite_guard(overwrite)
    from .datasets import Network_Data_MJS20 as ND
    from .utils import seconds_to_string as sec2str
    import time
    # Run routine to pickle Network_Data_MJS20 dataset.
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

def __pickle_roadNetCA(overwrite: bool=False):
    _overwrite_guard(overwrite)
    from .datasets import roadNetCA
    from .utils import seconds_to_string as sec2str
    import time
    # Run routine to pickle roadNetCA dataset.
    print('Loading in graph...')
    start: float = time.perf_counter()
    G: nx.Graph = roadNetCA['roadNet-CA']
    end: float = time.perf_counter()
    print(f'Loaded graph in {sec2str(end - start, rounding=2)}.')
    print(f'[1/1] Pickling {G.name.name}...')
    start: float = time.perf_counter()
    pickle_local_separators(G, overwrite=overwrite)
    end: float = time.perf_counter()
    print('Total pickling time:', sec2str(end - start, rounding=3))

def __pickle_stackoverflow(overwrite: bool=False):
    _overwrite_guard(overwrite)
    from .datasets import stackoverflow
    from .utils import seconds_to_string as sec2str
    import time
    # Run routine to pickle stackoverflow dataset.
    G: nx.Graph = stackoverflow['stackoverflow']
    print(f'[1/1] Pickling {G.name.name}...')
    start: float = time.perf_counter()
    pickle_local_separators(G, overwrite=overwrite)
    end: float = time.perf_counter()
    print('Total pickling time:', sec2str(end - start, rounding=3))

def __pickle_MajorOpenRoadNetworks(overwrite: bool=False):
    _overwrite_guard(overwrite)
    from .datasets import MajorOpenRoadNetworks
    from .utils import seconds_to_string as sec2str
    import time
    # Run routine to pickle Major_Road_Network_2018_Open_Roads.zip dataset.
    G: nx.Graph = MajorOpenRoadNetworks['Major_Road_Network_2018_Open_Roads']
    print(f'[1/1] Pickling {G.name.name}...')
    start: float = time.perf_counter()
    pickle_local_separators(G, overwrite=overwrite)
    end: float = time.perf_counter()
    print('Total pickling time:', sec2str(end - start, rounding=3))

def __pickle_infpower(overwrite: bool=False):
    _overwrite_guard(overwrite)
    from .datasets import infpower
    from .utils import seconds_to_string as sec2str
    import time
    # Run routine to pickle infpower dataset.
    G: nx.Graph = infpower['inf-power']
    print(f'[1/1] Pickling {G.name.name}...')
    start: float = time.perf_counter()
    pickle_local_separators(G, overwrite=overwrite)
    end: float = time.perf_counter()
    print('Total pickling time:', sec2str(end - start, rounding=3))

if __name__ == '__main__':
    overwrite: bool = False
    __pickle_infpower(overwrite=overwrite)
