# TODO:
# - add function docstrings

from os.path import commonpath
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import pandas as pd

PROJECT_ROOT: Path      = Path(__file__).parent.parent
SRC_ROOT: Path          = PROJECT_ROOT / 'src'
DATASETS_ROOT: Path     = PROJECT_ROOT / 'datasets'
MEDIA_ROOT: Path        = PROJECT_ROOT / 'media'
PICKLE_ROOT: Path       = PROJECT_ROOT / 'pickle'

def _mirror_datasets_folder_structure(path: Path):
    '''
        Mirrors the datasets folder structure recursively, rooted at
        the supplied path. This function is useful when creating the
        project top-level folders for dataset associated metadata
        (e.g. pickles, images).

        Parameters
        ----------
        path: Path
            Root from which to mirror the datasets folder.
    '''
    # Create the media folder if it doesn't exist.
    path.mkdir(exist_ok=True)
    # Mirror all datasets.
    dataset: Path
    for dataset in DATASETS_ROOT.iterdir():
        # Level of Network_Data_MJS20
        if not dataset.is_dir():
            continue
        mirror: Path = path / dataset.name
        mirror.mkdir(exist_ok=True)
        nested: Path
        for nested in dataset.iterdir():
            # Level of FoodWebs, etc.
            if not nested.is_dir():
                continue
            nested_mirror: Path = mirror / nested.name
            nested_mirror.mkdir(exist_ok=True)

def _doppelganger(directory: Path, path: Path, suffix: str) -> str:
    if not (path.is_file() and path.exists()):
        raise AttributeError('bruh:', path)
    prefix: str = commonpath((DATASETS_ROOT, path))
    ### +1 to remove leading '/'
    path_to_dataset: str = path.resolve().as_posix()[len(prefix)+1:]
    return (directory / path_to_dataset).with_suffix(suffix)

def _pickle_name(path: Path) -> Path:
    return _doppelganger(PICKLE_ROOT, path, '.pickle')

def _media_name(path: Path) -> Path:
    return _doppelganger(MEDIA_ROOT, path, '')

def _datasets_grouped_by_size(bins: int=10) -> Dict[float, List[Path]]:
    datasets: List[Tuple[Path, int]] = [
        (graph, graph.stat().st_size)
        for dataset in DATASETS_ROOT.iterdir()
        for folder in dataset.iterdir()
        for graph in folder.iterdir()
    ]
    sizes: List[int] = [y for x,y in datasets]
    bins = pd.qcut(sizes, q=bins)
    grouped_by_size: Dict[float, List[Path]] = {
        _bin.right: [] for _bin in bins
    }
    for graph, size in datasets:
        for prev, curr in zip(
                list(grouped_by_size.keys())[:-1],
                list(grouped_by_size.keys())[1:]
            ):
            if size <= prev:
                grouped_by_size[prev].append(graph)
            elif prev < size <= curr:
                grouped_by_size[curr].append(graph)
    return grouped_by_size

def _validate_graph_name(name: Any):
    '''
        Validates a graph's name.

        Parameters
        ----------
        name: Any
            Thing to check is a valid graph name.

        Raises
        ------
        TypeError
            If the graph's name attribute isn't a Path object.
    '''
    if not isinstance(name, Path):
        raise TypeError(f'expected name to be Path attribute, is in fact {type(name).__name__}: "{name}"')
