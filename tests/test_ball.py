from src.local_separators import (
    ball,
    Vertex,
)

from typing import (
    List,
    Tuple,
    Union
)

import networkx as nx
import unittest

ERROR_MSG: str = '\nVertex Is {vertex}, radius {radius}\n[ACTUAL] Vertices {actual.nodes} | Edges {actual.edges}\nVS\n[EXPECTED] Vertices {expected.nodes} | Edges {expected.edges}'

def err(vertex: Vertex, radius: Union[int, float], actual: nx.Graph, expected: nx.Graph) -> str:
    return ERROR_MSG.format(
        vertex=vertex, radius=radius, actual=actual, expected=expected
    )

class TestBall(unittest.TestCase):

    def test_path_graph(self):
        '''Tests local_separators.ball on path graphs'''
        # (test_name, n, v, r, expected_graph)
        params: List[Tuple[str, int, Vertex, Union[int, float], nx.Graph]] = [
            ('int radius < path length', 5, 2, 1, nx.path_graph((1, 2, 3))),
            ('int radius = path length', 5, 2, 4, nx.path_graph(5)),
            ('int radius > path length', 5, 2, 5, nx.path_graph(5)),
            ('half-int radius < path length / 2', 5, 2, 1.5, nx.path_graph((1, 2, 3))),
            ('central vertex, half-int radius = path length / 2', 5, 2, 2, nx.path_graph(5)),
            ('leaf vertex, half-int radius = path length / 2', 5, 0, 2, nx.path_graph(3)),
            ('central vertex, half-int radius > path length / 2', 5, 2, 2.5, nx.path_graph(5)),
            ('leaf vertex, half-int radius > path length / 2', 5, 0, 2.5, nx.path_graph(3)),
            ('half-int radius < path length', 5, 2, 1.5, nx.path_graph(3)),
        ]
        for test_name, n, v, r, expected in params:
            with self.subTest(msg=test_name):
                G: nx.Graph = nx.path_graph(n)
                actual: nx.Graph = ball(G, v, r)
                self.assertTrue(
                    nx.is_isomorphic(actual, expected),
                    msg=err(v, r, actual, expected)
                )

    def test_cycle_graph(self):
        '''Tests local_separators.ball on cycle graphs'''
        # (test_name, n, v, r, expected)
        params: List[Tuple[str, int, Vertex, Union[int, float], nx.Graph]] = [
            ('int radius < cycle length', 5, 2, 1, nx.path_graph((1, 2, 3))),
            ('int radius = floor(cycle length / 2)', 5, 0, 2, nx.path_graph(5)),
            ('int radius = cycle length', 5, 2, 5, nx.cycle_graph(5)),
            ('int radius > cycle length', 5, 2, 6, nx.cycle_graph(5)),
            ('half-int radius < cycle length / 2', 5, 2, 1.5, nx.path_graph(3)),
            ('half-int radius = odd cycle length / 2', 5, 2, 2.5, nx.cycle_graph(5)),
            ('half-int radius = even cycle length / 2', 6, 0, 3, nx.cycle_graph(6)),
            ('half-int radius > cycle length / 2', 5, 0, 3.5, nx.cycle_graph(5)),
        ]
        for test_name, n, v, r, expected in params:
            with self.subTest(msg=test_name):
                G: nx.Graph = nx.cycle_graph(n)
                actual: nx.Graph = ball(G, v, r)
                self.assertTrue(
                    nx.is_isomorphic(actual, expected),
                    msg=err(v, r, actual, expected)
                )

    def test_disconnected_graph(self):
        '''Testing the ball function on disconnected graphs.'''
        # (test_name, disconnected_graph, vertex, radius, expected_graph)
        with self.subTest(msg='isolated vertices graph'):
            vertex: Vertex = 0
            radius: Union[int, float] = 3
            G: nx.Graph = nx.Graph()
            G.add_nodes_from(range(5))
            actual: nx.Graph = ball(G, vertex, radius)
            expected: nx.Graph = nx.Graph()
            expected.add_node(vertex)
            self.assertTrue(
                nx.is_isomorphic(actual, expected),
                msg=err(vertex, radius, actual, expected)
            )

        with self.subTest(msg='isolated vertex, graph with 2 components'):
            vertex: Vertex = 5
            radius: Union[int, float] = 2.5
            G: nx.Graph = nx.complete_graph(4)
            G.add_node(vertex)
            actual: nx.Graph = ball(G, vertex, radius)
            expected: nx.Graph = nx.Graph()
            expected.add_node(vertex)
            self.assertTrue(
                nx.is_isomorphic(actual, expected),
                msg=err(vertex, radius, actual, expected)
            )
        
        with self.subTest(msg='connected component, graph with 2 components, one of them an isolated vertex'):
            vertex: Vertex = 0
            radius: Union[int, float] = 1
            G: nx.Graph = nx.complete_graph(4)
            G.add_node(5)
            actual: nx.Graph = ball(G, vertex, radius)
            expected: nx.Graph = nx.star_graph(3)
            self.assertTrue(
                nx.is_isomorphic(actual, expected),
                msg=err(vertex, radius, actual, expected)
            )

    def test_wheel_graph(self):
        with self.subTest(msg='wheel graph, radius 1'):
            vertex: Vertex = 0
            radius: Union[int, float] = 1
            G: nx.Graph = nx.wheel_graph(5)
            actual: nx.Graph = ball(G, vertex, radius)
            expected: nx.Graph = nx.star_graph(4)
            self.assertTrue(
                nx.is_isomorphic(actual, expected),
                msg=err(vertex, radius, actual, expected)
            )

        with self.subTest(msg='wheel graph, radius 1.5'):
            vertex: Vertex = 0
            radius: Union[int, float] = 1.5
            G: nx.Graph = nx.wheel_graph(5)
            actual: nx.Graph = ball(G, vertex, radius)
            expected: nx.Graph = G
            self.assertTrue(
                nx.is_isomorphic(actual, expected),
                msg=err(vertex, radius, actual, expected)
            )

        with self.subTest(msg='wheel graph, radius 2'):
            vertex: 0
            radius: Union[int, float] = 2
            G: nx.Graph = nx.wheel_graph(10)
            actual: nx.Graph = ball(G, vertex, radius)
            expected: nx.Graph = G
            self.assertTrue(
                nx.is_isomorphic(actual, expected),
                msg=err(vertex, radius, actual, expected)
            )

if __name__ == '__main__':
    unittest.main()
