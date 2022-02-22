'''
    pickle.py   -   
'''


from . import (
    _datasets_grouped_by_size,
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
    local_cutvertices: Dict[Vertex, int] = find_local_cutvertices(G)
    ### Make sure parent folder exists.
    graph_pickle.parent.mkdir(parents=True, exist_ok=True)
    ### Serialise result.
    with open(graph_pickle, 'wb') as handle:
        pickle.dump(local_cutvertices, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_dataset(dataset: str, overwrite: bool=False):
    '''
        This function calls pickle_local_separators for each dataset obtained
        via a dataset interface.

        Parameters
        ----------
        dataset: str
            Name of the dataset interface to connect to.
        overwrite: bool, default False

        Raises
        ------
        AttributeError
            If dataset isn't a correct dataset interface.
        ValueError
            If I can't code correctly.
    '''
    raise NotImplementedError('have you fixed wildcarding a dataset?')
    ### Obtain the dataset interface.
    P: Providence = Providence()
    D: DatasetInterface = next(
        filter(
            lambda x: x.file == dataset,
            P.interfaces
        )
    )
    ### Wildcard to obtain all datasets.
    datasets: List[Union[nx.Graph, GETITEM_RETURN_TYPE]] = D['*']
    ### Process the datasets in the dataset interface.
    # Define helper function that processes the potentially and
    # likely recursive data type.
    def _pickle_dataset(ds: Union[nx.Graph, GETITEM_RETURN_TYPE], overwrite: bool):
        '''
            Helper function that processes the recursive data structure
            and pickles all encountered datasets.

            Parameters
            ----------
            ds: Union[nx.Graph, GETITEM_RETURN_TYPE]
                Dataset which potentially consists of multiple nested
                graphs whose local cutvertices we need to pickle.
            overwrite: bool
                Whether or not we should overwrite a pre-existing pickle.
            
            Raises
            ------
            ValueError
                If the currently processed graph's name attribute isn't
                a Path object.
        '''
        if isinstance(ds, nx.Graph):
            # We simply have a Graph, pickle its local separators.
            print(f'Pickling {ds.name.name}...')
            start = time.time()
            pickle_local_separators(ds, overwrite)
            end = time.time()
            print(f'Done. Took {round(end - start, 1)} seconds.')
        else:
        # elif isinstance(ds, GETITEM_RETURN_TYPE):
            # We have to iterate through the values.
            v: Union[GETITEM_RETURN_TYPE, List[nx.Graph]]
            for v in ds.values():
                if isinstance(v, GETITEM_RETURN_TYPE):
                    pickle_local_separators(v, overwrite)
                elif isinstance(v, List[nx.Graph]):
                    for G in v:
                        pickle_local_separators(G, overwrite)
                else:
                    raise ValueError(f'v is of type {type(v).__name__}')
        # else:
        #     raise ValueError(f'ds is of type {type(ds).__name__}')
    # Call the helper function on the datasets contained within.
    dataset: Union[nx.Graph, GETITEM_RETURN_TYPE]
    for dataset in datasets:
        _pickle_dataset(dataset, overwrite)

def __pickle_Network_Neural_MJS20(overwrite: bool=False):
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

if __name__ == '__main__':
    __pickle_Network_Neural_MJS20(overwrite=True)
