from src.local_separators import (
    find_local_cutvertices,
    Vertex,
)

from typing import (
    Dict,
    List,
)

import networkx as nx
import pprint
import random
import unittest

class TestFindLocalCutvertices(unittest.TestCase):

    NUM_PARAMETRISED_TESTS: int = 40

    def test_path_graph(self):
        '''Tests the find_local_cutvertices function on a path graph'''
        params: List[int] = random.sample(
            range(3, 101),
            TestFindLocalCutvertices.NUM_PARAMETRISED_TESTS
        )
        num_vertices: int
        for num_vertices in params:
            G: nx.Graph = nx.path_graph(num_vertices)
            expected: Dict[Vertex, int] = {}
            actual: Dict[Vertex, int] = find_local_cutvertices(G)
            with self.subTest(msg=f'Path on {num_vertices} vertices'):
                self.assertEqual(actual, expected, msg=f'expected nothing, got:\n{pprint.pformat(actual)}')

    def test_cycle_graph(self):
        '''Tests the find_local_cutvertices function on a cycle graph'''
        params: List[int] = random.sample(
            range(4, 101),
            TestFindLocalCutvertices.NUM_PARAMETRISED_TESTS
        )
        num_vertices: int
        for num_vertices in params:
            G: nx.Graph = nx.cycle_graph(num_vertices)
            expected: Dict[Vertex, int] = {}
            actual: Dict[Vertex, int] = find_local_cutvertices(G)
            with self.subTest(msg=f'Cycle on {num_vertices} vertices'):
                self.assertEqual(actual, expected, msg=f'expected nothing, got:\n{pprint.pformat(actual)}')

    def test_complete_graph(self):
        '''Tests the find_local_cutvertices function on complete graphs'''
        params: List[int] = random.sample(
            range(5, 51),
            TestFindLocalCutvertices.NUM_PARAMETRISED_TESTS
        )
        num_vertices: int
        for num_vertices in params:
            G: nx.Graph = nx.complete_graph(num_vertices)
            expected: Dict[Vertex, int] = {}
            actual: Dict[Vertex, int] = find_local_cutvertices(G)
            with self.subTest(msg=f'Complete graph on {num_vertices} vertices'):
                self.assertEqual(actual, expected, msg=f'expected nothing, got:\n{pprint.pformat(actual)}')
