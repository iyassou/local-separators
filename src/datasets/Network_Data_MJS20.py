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
Network_Data_MJS20: DatasetInterface = DatasetInterface(
    __file__, _DATASET_EXTENSIONS, _edgelist_file_to_networkx_graph
)

sys.modules[__name__] = Network_Data_MJS20
