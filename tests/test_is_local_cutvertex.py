from src.local_separators import (
    is_local_cutvertex,
    Vertex,
)

from typing import (
    List,
    Tuple,
    Union,
)

import networkx as nx
import random
import unittest

class TestIsLocalCutvertex(unittest.TestCase):

    def test_path_graph(self):
        '''Tests the local_cutvertex function on path graphs.'''
        # (test_name, num_vertices, vertex, radius, expected)
        params: List[Tuple[str, int, Vertex, Union[int, float], bool]] = [
            ('central vertex, even radius', 101, 50, 20, True),
            ('central vertex, odd radius', 9, 5, 2, True),
            ('central vertex, infinite radius', 101, 50, float('inf'), True)
        ]

        for test_name, num_vertices, vertex, radius, expected in params:
            with self.subTest(msg=test_name):
                G: nx.Graph = nx.path_graph(num_vertices)
                actual: bool = is_local_cutvertex(G, vertex, radius)
                self.assertEqual(
                    actual,
                    expected,
                    msg=f'expected {expected}, got {actual}'
                )

    def test_cycle_graph(self):
        '''Tests the local_cutvertex function on cycle graphs.'''
        # (test_name, num_vertex, vertex, radius, expected)
        params: List[Tuple[str, int, Vertex, Union[int, float], bool]] = [
            ('int radius < cycle length', 5, 0, 2, True),
            ('int radius = floor(cycle length / 2)', 5, 0, 4, True),
            ('int radius = cycle length', 5, 0, 10, False),
            ('int radius > cycle length', 5, 0, 12, False),
            ('int radius = inf', 5, 0, float('inf'), False),
            ('half-int radius < cycle length / 2', 5, 0, 3, True),
            ('half-int radius = odd cycle length / 2', 5, 0, 5, False),
            ('half-int radius = even cycle length / 2', 5, 0, 6, False),
            ('half-int radius > cycle length / 2', 5, 0, 7, False)
        ]

        for test_name, num_vertices, vertex, radius, expected in params:
            with self.subTest(msg=test_name):
                G: nx.Graph = nx.cycle_graph(num_vertices)
                actual: bool = is_local_cutvertex(G, vertex, radius)
                self.assertEqual(
                    actual,
                    expected,
                    msg=f'expected {expected}, got {actual}'
                )

    def test_complete_graph(self):
        '''Tests the local_cutvertex function on complete graphs.'''
        sub_tests: int = 50
        min_n: int = 30
        max_n: int = 100

        N: List[int] = random.sample(range(min_n, max_n), sub_tests)

        n: int
        for n in N:
            vertex: Vertex = random.choice(range(n))
            K_n: nx.Graph = nx.complete_graph(n)
            with self.subTest(msg=f'K_{{{n}}} | Vertex: {vertex} | Half-Radius 1'):
                actual: bool = is_local_cutvertex(K_n, vertex, 2)
                self.assertTrue(actual)
            with self.subTest(msg=f'K_{{{n}}} | Vertex: {vertex} | Half-Radius ≥ 1'):
                actual: bool = is_local_cutvertex(K_n, vertex, 2 + random.choice(range(1, n)))
                self.assertFalse(actual)
