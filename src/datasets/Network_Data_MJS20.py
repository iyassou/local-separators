from . import (
    DatasetInterface,
    _edgelist_file_to_networkx_graph,
)

from typing import (
    Set,
    Tuple,
)

import sys

_DATASET_EXTENSIONS: Set[str] = {'.dat'}
_PROBLEMATIC_DATASETS: Tuple[str] = tuple(
    f'rat_brain_{i+1}.dat' for i in range(3)
)
Network_Data_MJS20: DatasetInterface = DatasetInterface(
    __file__, _DATASET_EXTENSIONS, _PROBLEMATIC_DATASETS, _edgelist_file_to_networkx_graph
)

sys.modules[__name__] = Network_Data_MJS20
