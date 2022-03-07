'''
    local_separators.py -   Functions for identifying local 1- and 2-separators,
                            as well as related structures inside of graphs.
'''

from math import (
    ceil,
    floor,
    modf,
)
from typing import (
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Set,
    TypeVar,
    Tuple,
    Union,
)

import networkx as nx
import re

Vertex = TypeVar('Vertex')
NAMING_CONVENTION_REGEX = re.compile(r'[A-Za-z0-9]+\_{[\d]+}')

class LocalCutvertex(NamedTuple):
    '''
        Given a graph :math:`G` and :math:`r\in\mathbb{N}\cup\left\lbrace\infty\right\rbrace`, math:`v\in V(G)` is an
        :math:`r`-local cutvertex if the ball of radius :math:`\frac{r}{2}` with :math:`v` removed is disconnected.

        If :math:`v` is a local cutvertex, then :math:`B_{\frac{r}{2}}(v)-v` contains :math:`c>1` components, and so
        :math:`B_{\frac{r}{2}}(v)` contains edges going from :math:`v` to the vertices in these :math:`c` components,
        effectively forming a partition of :math:`E_G(v)` in :math:`G`.
        
        We keep track of this edge-partition as well as the radius so as to enable splitting at a local cutvertex, an
        operation detailed below. Note that we only need to store :math:`u\in V(G)` for :math:`uv\in E_G(v)` since it
        is implicit that the edge's other end is in fact :math:`v`.

        vertex: Vertex
            The vertex in question, i.e. its identifier in the graph G.
        locality: int
            The locality of the local cutvertex.
        edge_partition: Set[Tuple[Vertex, ...]]
            Partition of :math:`E_G(v)` according to the components of :math:`B_{\frac{r}{2})(v)-v`.
    '''
    vertex: Vertex
    locality: int
    edge_partition: Set[Tuple[Vertex, ...]]

def ball(G: nx.Graph, v: Vertex, r: Union[int, float]) -> nx.Graph:
    '''
        Returns a view of the ball of radius r around the vertex v in the graph G.

        Parameters
        ----------
        G: nx.Graph
        v: Vertex
        r: Union[int, float]

        Notes
        -----
        When the radius is an integer, the ball is the "induced subgraph of G whose
        vertices are those of distance at most r from v and without all edges
        joining two vertices of distance precisely r".
        When the radius is a half-integer, the ball is the "induced subgraph of G
        whose vertices are those of distance at most r from v".

        Raises
        ------
        ValueError
            If r is neither an integer nor a half-integer.

        Returns
        -------
        nx.Graph
    '''
    # Begin by making sure that the radius conforms to our expectations.
    decimal, integral = modf(r)
    if decimal and decimal != 0.5:
        raise ValueError(f'expected integer or half-integer radius, got: "{r}"')
    # Compute the shortest paths from the v to all other vertices in the graph G
    # that are at most r away from v.
    # NOTE: although sssp supports passing a float to cutoff, it doesn't result
    # in the expected behaviour i.e. cutoff=1.5 includes paths of length 2 for
    # some reason that I do not understand from reading the underlying source
    # code, hence why I'm using the integral part of the radius, which results in
    # the expected behaviour. This doesn't interfere with passing float('inf')
    # as the radius, hence this is a, if not the, sensible fix.
    d: Dict[Vertex, List[Vertex]] = nx.algorithms.shortest_paths.unweighted.single_source_shortest_path(
        G, v, cutoff=integral
    )
    # To obtain the ball, we need to filter out the vertices and edges to include from G.
    def filtvert(x: Vertex) -> bool:
        # The vertices in the ball are those with distance at most r from v.
        return x in d
    def filtedge(x: Vertex, y: Vertex) -> bool:
        # The edges xy in the ball are those that satisfy the relation:
        #       d_G(v,x) + d_G(v,y) + 1 ≤ 2 * r
        # Recall: A path on n vertices has length n-1.
        # If the end of an edge isn't in the dictionary of shortest paths from
        # v that have length at most r, then we can't keep that end, and hence
        # cannot keep the edge.
        if not G.has_edge(x, y):
            # The ball is an induced subgraph, hence if the edge isn't present
            # in the original graph, don't keep it.
            return False
        try:
            d_vx = len(d[x]) - 1
            d_vy = len(d[y]) - 1
            return d_vx + d_vy + 1 <= 2 * r
        except KeyError:
            return False
    # Return the induced subgraph view filtering using filtvert and filtedge.
    return nx.classes.graphviews.subgraph_view(G, filter_node=filtvert, filter_edge=filtedge)

def is_local_cutvertex(G: nx.Graph, v: Vertex, r: int) -> bool:
    '''
        Is v is an r-local cutvertex in G?

        Parameters
        ----------
        G : nx.Graph
        v: Vertex
        r: int

        Notes
        -----
        Given a parameter :math:`r\in\mathbb{N}\cup\left\lbrace\infty\right\rbrace`,
        a vertex :math:`v` is an :math:`r`-local cutvertex if it separates the ball
        of radius :math:`r/2` around :math:`v`; formally: :math:`B_{r/2}(v)-v` is
        disconnected.
        In Python3, float('inf') / 2 == float('inf') so we don't need to give
        float('inf') any special attention here.

        Returns
        -------
        bool
    '''
    # Obtain the ball of radius r/2 centered at v in G.
    # Remove the vertex v from the ball.
    B: nx.Graph = nx.classes.graphviews.subgraph_view(
        ball(G, v, r/2), filter_node=lambda x: x != v
    )
    # Is the ball disconnected?
    return not nx.algorithms.components.is_connected(B)

def find_local_cutvertices(G: nx.Graph, min_locality: int=3) -> List[LocalCutvertex]:
    '''
        Iterates through graph vertices and detects r-local cutvertices. This algorithm has
        time complexity :math:`O(n\cdot\log(k))` where :math:`n` is the number of vertices,
        and :math:`k` is the size of the largest component of :math:`G`.

        Parameters
        ----------
        G: nx.Graph
        min_locality: int, default 3

        Notes
        -----
        The maximum radius in the binary search is taken to be the number of vertices in the connected
        component of the vertex under consideration in the search loop.
        The value of r associated with a vertex is the maximal value of r for which that vertex
        is an r-local cutvertex.
        Since ":math:`v` an :math:`(r+1)`-local cutvertex" :math:`\implies` ":math:`v` an :math:`r`-local cutvertex",
        the maximal value of r is found using a binary search.
        This algorithm only picks up strictly local cutvertices i.e. no component-level separators.

        Returns
        -------
        List[LocalCutvertex]
            A list of :math:`k` local cutvertices, where their respective radii :math:`r_i` with :math:`i\in\left\lbrace 1,\ldots,k\right\rbrace`
            are the largest for which they're :math:`r_i`-local cutvertices respectively in the graph :math:`G`.
    '''
    # To determine if a vertex is indeed a local separator and not a component-level
    # separator, we'll need the connected components in the graph.
    components: List[Set[Vertex]] = list(nx.connected_components(G))
    # Prepare the return value list.
    local_cutvertices: List[LocalCutvertex] = []
    # Iterate through each vertex in the graph.
    for v in G.nodes:
        # Prepare the binary search bounds.
        mi: int = min_locality
        component: Set[Vertex] = next(comp for comp in components if v in comp)
        ma: int = len(component)
        if mi > ma:
            # This can occur if there are fewer vertices in the
            # component v belongs to than expected. In this case,
            # the radius we would potentially determine for v is
            # not of interest, given that it would be smaller than
            # min_radius, hence we should move onto the next vertex.
            continue
        mid: int = None
        v_is_a_local_cutvertex: bool = None
        # Proceed with the binary search.
        while True:
            ### Update the current locality.
            if ma - mi == 1 and v_is_a_local_cutvertex is not None:
                # One of either the mi or ma values at the previous iteration of
                # this algorithm was already tried, in which case the typical
                # process of computing a new value of mid would create an infinite
                # loop, hence the intervention for this special case.
                if not v_is_a_local_cutvertex:
                    # GUARANTEES:       - ma corresponds to the previously tried value.
                    #                   - v IS NOT a ma-local cutvertex.
                    # PLAN OF ACTION:   Determine if v is a mi-local cutvertex.
                    #                   Update mid = mi, regardless of whether v is
                    #                   a mid-local cutvertex.
                    v_is_a_local_cutvertex: bool = is_local_cutvertex(G, v, mi)
                    mid: int = mi
                    break
                else:
                    # GUARANTEES:       - mi corresponds to the previously tried value.
                    #                   - v IS a mi-local cutvertex.
                    # PLAN OF ACTION:   Determine if v is a ma-local cutvertex.
                    #
                    #                   If v is a ma-local cutvertex, then update mid = ma.
                    #
                    #                   If v is not a ma-local cutvertex, then let mid stay
                    #                   as mi, and revert v_is_a_local_cutvertex to True.
                    v_is_a_local_cutvertex: bool = is_local_cutvertex(G, v, ma)
                    if v_is_a_local_cutvertex:
                        mid: int = ma
                    else:
                        v_is_a_local_cutvertex: bool = True
                    break
            else:
                # Since is_local_cutvertex only accepts integers we're taking the floor.
                mid: int = mi + floor((ma - mi) / 2)
            ### Check if v is mid-local cutvertex.
            v_is_a_local_cutvertex: bool = is_local_cutvertex(G, v, mid)
            ### Check if the search bounds have crossed.
            if mi == ma:
                break
            ### Update the binary search boundaries otherwise.
            if v_is_a_local_cutvertex:
                # v is a mid-local cutvertex, so check if this is the maximal
                # radius by lowering the lower bound.
                mi: int = mid
            else:
                # v is not a mid-local cutvertex, so check if there is a
                # smaller radius for which v is an r-local cutvertex by
                # increasing the upper bound.
                ma: int = mid
        # End of the binary search. Is v a genuine mid-local cutvertex (i.e.
        # v is a mid-local separator that doesn't separate its component)?
        if v_is_a_local_cutvertex:
            component_without_v: nx.Graph = nx.classes.graphviews.subgraph_view(
                G, filter_node=lambda x: x in component and x != v
            )
            # Run the component connectedness check.
            if nx.algorithms.components.is_connected(component_without_v):
                # v does not separate its component, hence it's a genuine local separator.

                # Before appending v to the list of local cutvertices we need to obtain the
                # components of B_{r/2}(v)-v and track the edges they send to v in G.

                # Step 1: Obtain the punctured ball of radius r/2 around v.
                punctured_ball: nx.Graph = nx.classes.graphviews.subgraph_view(
                    ball(G, v, mid/2), filter_node=lambda x: x != v
                )
                # Step 2: Obtain the connected components of the punctured ball.
                punctured_ball_components: Generator[Set[Vertex], None, None] = nx.connected_components(punctured_ball)
                # Step 3: Obtain the neighbourhood of v in G.
                neighbourhood: Set[Vertex] = set(G.neighbors(v))
                # Step 4: Obtain the intersections of the neighbourhood of v in G with
                #         the components of the punctured ball.
                edge_partition: Set[Tuple[Vertex, ...]] = set(
                    tuple(neighbourhood.intersection(comp)) for comp in punctured_ball_components
                )

                # Add LocalCutvertex v to the list of local cutvertices.
                local_cutvertices.append(
                    LocalCutvertex(vertex=v, locality=mid, edge_partition=edge_partition)
                )
    # Done iterating over the vertices.
    return local_cutvertices

def _split_naming_convention(v: Vertex, identifier: int) -> str:
    '''
        Returns the naming convention for a copy of the vertex v
        that results from splitting a graph at v.

        Parameters
        ----------
        v: Vertex
            The vertex we're splitting at i.e. creating a copy of.
        identifier: int
            An identifier for this specific copy.

        Notes
        -----
        The naming convention for the copies is that of LaTeX subscripts
        appended to the original vertex name numbered from 1 to
        :math:`d_G(v)`, where G is the graph the vertex is present in.

        Returns
        -------
        str
    '''
    return f'${str(v).replace("$", "")}_{{{str(identifier)}}}$'

def split_at_local_cutvertices(G: nx.Graph, local_cutvertices: List[LocalCutvertex], inplace: bool=False) -> Optional[nx.Graph]:
    '''
        Splits a vertex v in a graph and numbers the split vertices by the components of
        the ball of radius r/2 with the vertex v removed.

        Parameters
        ----------
        G: nx.Graph
        local_cutvertices: List[LocalCutvertices]
            The list of local cutvertices to split at.
        inplace: bool, default False
            Whether or not this operation should be done in place. If False, a copy
            of the original graph is operated on and returned at the end.
        
        Notes
        -----
        Splitting at a vertex :math:`v` with a radius :math:`r` equates to supplying the
        components of the :math:`B_{\frac{r}{2}}(v)-v` with unique copies of the vertex, and
        subsequently adding edges between :math:`v` and its numbered copies.

        Although this is correct and not error-prone for splitting at a single local cutvertex,
        obtaining the punctured balls while also introducing split vertices as we split at
        multiple local cutvertices is error-prone, as the introduction of the split vertices
        would modify the underlying graph.
        Specifically, an issue that can arise is if a local cutvertex :math:`v` is in the
        punctured ball of another local cutvertex :math:`w`, then splitting at :math:`v` first
        would modify the ball we'd obtain around :math:`w` because of the newly introduced
        split vertices.
        To remedy this, we instead introduce split vertices by operating on the edge-partition
        of each local cutvertex, which is a standalone approach in that it doesn't affect the
        other local cutvertices.

        The split vertices are given names according to _split_naming_convention, where
        identifier is an increasing integer index, starting at 1.
        Split vertices are tagged with a 'split' attribute set to True.

        Returns
        -------
        Optional[nx.Graph]
            A graph is returned if the operation isn't done in place.
    '''
    # Which graph are we operating on?
    if not inplace:
        graph: nx.Graph = G.copy()
    else:
        graph: nx.Graph = G
    # Construct the dictionary that will give each split vertex the 'split' attribute.
    attr: Dict[Vertex, bool] = dict()
    # Keep a set of all available local cutvertices.
    lcv_vertices: Set[Vertex] = set(lcv.vertex for lcv in local_cutvertices)
    # Construct the split vertices using the edge-partition.
    lcv: LocalCutvertex
    for lcv in local_cutvertices:
        # Go through the subsets in the edge-partition.
        subset: Tuple[Vertex, ...]
        for i, subset in enumerate(lcv.edge_partition):
            # Modify the subset to exclude other local cutvertices.
            subset: Tuple[Vertex, ...] = tuple(
                vertex for vertex in subset
                if vertex not in lcv_vertices
            )
            if not subset:
                continue
            # Create the split vertex first.
            split_vertex: Vertex = _split_naming_convention(lcv.vertex, i+1)
            # We want to remove the edges from the local cutvertex to the vertices in this subset,
            graph.remove_edges_from(
                (lcv.vertex, neighbour) for neighbour in subset
            )
            # and add edges to the split vertex, named using _split_naming_convention,
            graph.add_edges_from(
                (neighbour, split_vertex) for neighbour in subset
            )
            # before finally adding the edge from the split vertex to the local cutvertex.
            graph.add_edge(split_vertex, lcv.vertex)
            # Modify the split vertex attribute.
            attr[split_vertex] = True
    # Distinguish the split vertices.
    nx.set_node_attributes(graph, attr, name='split')
    # Return the graph if necessary.
    if not inplace:
        return graph
