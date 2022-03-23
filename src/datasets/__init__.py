# TODO:
# - add Hamming-distance aware search for find_dataset_path? [LOL]
# - improve code for wildcard matching in DatasetInterface

from .. import _doppelganger

from functools import lru_cache
from os.path import commonpath
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import importlib
import networkx as nx
import pickle

PROJECT_ROOT: Path          = Path(__file__).parent.parent.parent
SHORTEST_PATHS_ROOT: Path   = PROJECT_ROOT / 'shortest_paths'

def _shortest_paths_name(path: Path) -> Path:
    return _doppelganger(SHORTEST_PATHS_ROOT, path, '.shortest_paths')

GETITEM_RETURN_TYPE: type = Dict[str, Union['GETITEM_RETURN_TYPE', List[nx.Graph]]]

def _find_dataset_path(dataset: str) -> Union[Path, List[Path]]:
    '''
        Given a dataset name, this function finds the corresponding
        dataset.

        Parameters
        ----------
        dataset: str
            Name of the dataset to look for.
        
        Returns
        -------
        Path
            Path object to the dataset.
    '''
    matches: List[Path] = list(DATASETS_ROOT.rglob(dataset))
    if not len(matches):
        raise LookupError(f'could not find dataset "{dataset}"')
    if len(matches) == 1:
        return matches[0]
    return matches
    # name: str
    # extension: Optional[str] = None
    # try:
    #     name, extension = dataset.split('.')
    # except ValueError:
    #     name = dataset

def _edge_from_line(line: str) -> Optional[Tuple[int, int]]:
    '''
        Utility function to create an edge from a .dat line.
        
        Parameters
        ----------
        line : str

        Notes
        -----
        The edges in most edgelists are in the format 'A B' which corresponds to a
        directed edge from B to A. I'm interpreting it as an edge from A to B.
        If the line is an empty line, then the function returns None.
        
        Returns
        -------
        Optional[Tuple[int, int]]
    '''
    if not line.strip(): # Empty line
        return None
    if line.startswith('#') or line.startswith('%'): # Comment
        return None
    return tuple(map(int, line.split()))

def _edgelist_file_to_networkx_graph(file: Path) -> nx.Graph:
    '''
        Converts an edgelist file to a networkx Graph.

        Parameters
        ----------
        file: Path
            Path to the edgelist file

        Returns
        -------
        nx.Graph
    '''
    with open(file, 'r') as f:
        edges: Iterator[Tuple[int, int]] = map(
            _edge_from_line,
            f.readlines()
        )
    return nx.Graph(
        filter(lambda x: x is not None, edges), # We filter None from the edgelist.
        name=Path(file)
    )

class DatasetInterface:
    '''
        This class acts as a generic interface to access the datasets
        under the same location. It is meant to be subclassed.
        Invididual datasets are accessed via __getitem__ calls directly
        on the class, i.e. Network_Data_MJS20['FoodWebs'].
        To achieve this a singleton will be instantiated and loaded
        into sys.modules.
    '''
    WILDCARD: str = '*'

    def __init__(self,
                 file: str,
                 dataset_extensions: Set[str],
                 problematic: Tuple[str],
                 dataset_parser: callable):
        '''
            Initialisation function.

            Parameters
            ----------
            file: str
                Name of the subclass' file.
            dataset_extensions: Set[str]
                Set of extensions that correspond to an individual dataset.
            problematic: Tuple[str]
                Tuple of datasets which are problematic to parse and so should
                be skipped. These datasets should be given with their file
                extension.
            dataset_parser: callable, default _edgelist_file_to_networkx_graph
                Function which given a dataset file (e.g. data.dat) parses
                it into an nx.Graph.

            Notes
            -----
            Convention is mirroring the dataset names in their associated
            Python files.
            Since some of these networks are rather large, I'm electing to
            use a cache around __getitem__ calls, that way only specific
            datasets are loaded when necessary and memory usage shouldn't
            be too bad.
            Cache size is the total number of datasets and the total number
            of nested folders, minus the number of problematic datasets.
        '''
        ### Store file.
        self.file = Path(file).stem
        ### Store dataset extensions.
        self.dataset_extensions: Set[str] = dataset_extensions
        ### Store problematic datasets.
        self.problematic: Tuple[str] = tuple(problematic)
        ### Store dataset parser.
        self.dataset_parser: callable = dataset_parser
        ### Determine the maximum cache size.
        # Find the number of datasets.
        num_datasets: int = sum(
            dataset.name not in self.problematic
            for ext in self.dataset_extensions
            for dataset in self.dataset_folder().rglob(f'*{ext}')
        )
        # Find the number of folders.
        num_folders: int = sum(
            x.is_dir() for x in self.dataset_folder().rglob('*')
        )
        # Find the number of problematic datasets.
        num_problematic: int = len(self.problematic)
        # Maximum cache size is then...
        MAX_CACHE_SIZE: int = num_datasets + num_folders - num_problematic
        ### Decorate the __getitem__ magic method.
        self.__getitem__ = lru_cache(maxsize=MAX_CACHE_SIZE)(self.__getitem__)

    def dataset_folder(self) -> Path:
        '''
            Returns the Path object corresponding to the main dataset
            root in the datasets folder.

            Returns
            -------
            Path
        '''
        return DATASETS_ROOT / self.file
    
    def __getitem__(self, attr: str) -> Union[nx.Graph, List[Union[nx.Graph, GETITEM_RETURN_TYPE]]]:
        '''
            Access nx.Graph datasets contained under the main dataset folder.

            Parameters
            ----------
            attr: str
                Name of the dataset or list of datasets to access

            Notes
            -----
            If attr is an individual dataset, we're returning an nx.Graph.
            If attr is a folder with nothing but individual datasets, we're
            return a List[nx.Graph].
            If attr is a folder with individual datasets and nested folders,
            we're returning a List[nx.Graph, GETITEM_RETURN_TYPE].
            
            Raises
            ------
            AttributeError
                When the requested attr is neither a folder nor dataset.

            Returns
            -------
            Union[nx.Graph, List[Union[nx.Graph, GETITEM_RETURN_TYPE]]
        '''
        # Wildcard selects all available datasets.
        if attr == self.WILDCARD:
            raise NotImplementedError
        '''
            all_datasets: List[Union[nx.Graph, GETITEM_RETURN_TYPE]] = list()
            knick_knack: Path
            for knick_knack in self.dataset_folder().rglob('*'):
                if knick_knack.is_dir():
                    all_datasets.append(
                        # [
                            {item.name: self[item.name]} if item.is_dir() else self.dataset_parser(item)
                            for item in filter(
                                lambda x: x.is_dir() or x.suffix in self.dataset_extensions and x.name not in self.problematic,
                                knick_knack.iterdir()
                            )
                        # ]
                    )
                elif knick_knack.suffix in self.dataset_extensions:
                    if knick_knack.name in self.problematic:
                        continue
                    all_datasets.append(self.dataset_parser(knick_knack))
            return all_datasets
        '''
        # Cycle through all allowed extensions and all nested folders.
        knick_knack: Path
        for knick_knack in self.dataset_folder().rglob('*'):
            if knick_knack.is_dir() and knick_knack.name == attr:
                # items is a list of [nested folders | files with dataset extensions]
                items: List[Path] = [
                    item for item in knick_knack.iterdir()
                    if item.is_dir() or item.suffix in self.dataset_extensions
                ]
                datasets: List[Union[Dict[str, List[nx.Graph]], nx.Graph]] = []
                item: Path
                for item in items:
                    if item.is_dir():
                        datasets.append(
                            { item.name: self[item.name] }
                        )
                    elif item.suffix in self.dataset_extensions:
                        if item.name in self.problematic:
                            continue
                        datasets.append(self[item.stem])
                return datasets
            elif knick_knack.suffix in self.dataset_extensions:
                # Not a directory, dealing with individual files here.
                if knick_knack.name in self.problematic:
                    continue
                if knick_knack.stem == attr:
                    # Found the dataset.
                    G: nx.Graph = self.dataset_parser(knick_knack)
                    # Associate the shortest_paths data.
                    graph_shortest_paths: Path = _shortest_paths_name(G.name)
                    if graph_shortest_paths.exists() and graph_shortest_paths.stat().st_size:
                        with open(graph_shortest_paths, 'rb') as handle:
                            G.graph['shortest_paths'] = pickle.load(handle)
                    else:
                        graph_shortest_paths.parent.mkdir(parents=True, exist_ok=True)
                        # nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path
                        shortest_paths: dict = nx.all_pairs_shortest_path(G)
                        with open(graph_shortest_paths, 'wb') as handle:
                            pickle.dump(shortest_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        G.graph['shortest_paths'] = shortest_paths
                    return G
        # attr is neither a dataset nor a folder, sound the alarm
        raise AttributeError(f'No dataset or folder named "{attr}"')

    def __str__(self) -> str:
        return f'DatasetInterface<"{self.file}">'

class Providence:
    '''
        Interface over all datasets in the datasets folder.
        Works by holding all defined interfaces and iteratively
        searching through them.
    '''
    def __init__(self):
        '''
            Initialisation function that creates DatasetInterfaces
            for all available top-level datasets. Note that:

                import src.datasets.DATASET
            
            will import the singleton class that's instantiated by
            convention in the Python file.
        '''
        ### Import all available, implemented DatasetInterfaces.
        dataset_interfaces_dir: Path = SRC_ROOT / 'datasets'
        self.interfaces: List[DatasetInterface] = [
            importlib.import_module(f'src.datasets.{file.stem}')
            for file in dataset_interfaces_dir.glob('*.py')
            if not file.stem.startswith('__')
        ]
        
    def __getitem__(self, attr: str) -> Union[nx.Graph, List[Union[nx.Graph, GETITEM_RETURN_TYPE]]]:
        '''
            Search function that gives access to all available datasets
            for which a Python wrapper is implemented.

            Parameters
            ----------
            attr: str
                Name of the dataset or folder to look for.

            Notes
            -----
            If attr is an individual dataset, we're returning an nx.Graph.
            If attr is a folder with nothing but individual datasets, we're
            return a List[nx.Graph].
            If attr is a folder with individual datasets and nested folders,
            we're returning a List[nx.Graph, GETITEM_RETURN_TYPE].
            
            Raises
            ------
            AttributeError
                When the requested attr is neither a folder nor dataset.

            Returns
            -------
            Union[nx.Graph, List[Union[nx.Graph, GETITEM_RETURN_TYPE]]
        '''
        ### First check if attr is an interface name.
        for interface in self.interfaces:
            if interface.file == attr:
                return interface[interface.WILDCARD]
        ### Cycle through all interfaces and look for attr.
        for interface in self.interfaces:
            try:
                return interface[attr]
            except AttributeError:
                continue
        raise AttributeError(f'could not locate dataset/folder "{attr}"')
