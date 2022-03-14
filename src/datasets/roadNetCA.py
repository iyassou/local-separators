from . import (
    DatasetInterface,
    _edgelist_file_to_networkx_graph,
)

from typing import (
    Set,
    Tuple,
)

import sys

_DATASET_EXTENSIONS: Set[str] = {'.txt'}
_PROBLEMATIC_DATASETS: Tuple[str] = ()
roadNetCA: DatasetInterface = DatasetInterface(
    __file__, _DATASET_EXTENSIONS, _PROBLEMATIC_DATASETS, _edgelist_file_to_networkx_graph
)

sys.modules[__name__] = roadNetCA
