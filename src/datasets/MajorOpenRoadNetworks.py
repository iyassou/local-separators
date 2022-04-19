from . import DatasetInterface
from ..utils import path_to_str

from pathlib import Path
from typing import (
    List,
    Tuple,
    Set,
)

import networkx as nx
import shapefile
import sys

def shapefile_stats(file: Path):
    with shapefile.Reader(path_to_str(file)) as shp:
        pass

def parse_MajorOpenRoadNetworks_dataset(file: Path) -> nx.Graph:
    '''
        Parses the Major_Road_Network_2018_Open_Roads.zip dataset.
    '''
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
        # print(shp.shapeRecord(1).shape.parts)
        # for i, record in enumerate(shp.shapeRecords()):
        #     if record.record['roadClas_1'] == 'A68':
        #         print('RECORD NUMBER:', i)
        #         print('RECORD SHAPE TYPE:', record.shape.shapeTypeName)
        #         print('RECORD SHAPE POINTS:', record.shape.points)
        #         for field_name, field_type, _, __ in shp.fields:
        #             if field_type != 'C' or field_name == 'DeletionFlag':
        #                 continue
        #             attr = record.record[field_name]
        #             if not attr:
        #                 continue
        #             print(f'{field_name} = {attr}')
        #         print('='*30)
    # print('Total points read:', index)
    # Return the graph.
    return G

_DATASET_EXTENSIONS: Set[str] = {'.zip'}
_PROBLEMATIC_DATASETS: Tuple[str] = ()
MajorOpenRoadNetworks: DatasetInterface = DatasetInterface(
    __file__, _DATASET_EXTENSIONS, _PROBLEMATIC_DATASETS, parse_MajorOpenRoadNetworks_dataset
)

sys.modules[__name__] = MajorOpenRoadNetworks
