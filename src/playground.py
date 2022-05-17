'''
    playground.py   -   Code used to mainly test visualise.py
'''

from . import (
    _pickle_name,
    PROJECT_ROOT
)

from .datasets import (
    Network_Data_MJS20 as NDMJS20,
    roadNetCA,
    infpower,
    stackoverflow,
    MajorOpenRoadNetwork,
)
from .local_separators import (
    Vertex,
    LocalCutvertex,
    is_local_cutvertex,
    split_at_local_cutvertices,
    ball,
)
from .utils import (
    seconds_to_string as sec2str,
    bounding_box_2d,
    escape_underscore,
    latex_safe_string as lss,
    pluralise,
    collinear,
    nearest,
    path_to_str,
    FigureSize, A0, A1, A2, A3, A4,
    visually_distinct_colours,
    euclidean_distance,
    polygon,
)
from .visualise import (
    draw_graph,
    draw_local_cutvertices,
    draw_locality_heatmap,
    draw_split_vertices,
    calculate_marker_sizes,
)

from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import humanize
import itertools
import math
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import operator
import pickle
import random
import shapefile
import tabulate
import time

# UTILITIES

def __get_Network_Data_MJS20_graphs() -> List[nx.Graph]:
    categories: List[str] = 'FoodWebs Genetic Language Metabolic Neural Social Trade'.split()
    datasets: List[List[nx.Graph]] = list(map(lambda x: NDMJS20[x], categories))
    graphs: List[nx.Graph] = [graph for dataset in datasets for graph in dataset]
    return graphs

def __get_NDMJS20_no_local_cutvx() -> List[nx.Graph]:
    graphs: List[nx.Graph] = __get_Network_Data_MJS20_graphs()
    def no_local_cutvx(G: nx.Graph) -> bool:
        with open(_pickle_name(G.name), 'rb') as handle:
            lcvx: List[LocalCutvertex] = pickle.load(handle)
        return not lcvx
    return list(filter(no_local_cutvx, graphs))

def __get_W13_special(hub: int=1) -> nx.Graph:
    G: nx.Graph = nx.cycle_graph(range(hub+1, hub+13))
    spokes = []
    for i in range(hub+1, hub+13, 4):
        spokes.extend([(hub, i), (hub, i+1)])
    G.add_edges_from(spokes)
    return G

def __get_MORN() -> nx.Graph:
    return MajorOpenRoadNetwork['Major_Road_Network_2018_Open_Roads']

def __get_pickled_local_cutvertices(G: nx.Graph) -> List[LocalCutvertex]:
    '''
        Attempt to retrieve the pickled local cutvertices of a graph, using
        its name and pickle name.

        Parameters
        ----------
        G: nx.Graph

        Raises
        ------
        FileNotFoundError
            If the pickled local cutvertices file doesn't exist.

        Returns
        -------
        List[LocalCutvertex]
            The list of pickled local cutvertices, if they exist.
    '''
    pickle_file: Path = _pickle_name(G.name)
    if not pickle_file.exists():
        raise FileNotFoundError(f"couldn't find pickle file for {G.name.stem}")
    with open(pickle_file, 'rb') as handle:
        local_cutvertices: List[LocalCutvertex] = pickle.load(handle)
    return local_cutvertices

def __get_infpower_kamada_kawai_layout() -> Dict[Vertex, np.ndarray]:
    layout: Path = PROJECT_ROOT / 'infpower' / 'kamada_kawai.pickle'
    if layout.exists():
        with open(layout, 'rb') as handle:
            return pickle.load(handle)
    layout.parent.mkdir(exist_ok=True)
    with open(layout, 'wb') as handle:
        pos = nx.kamada_kawai_layout(infpower['inf-power'])
        pickle.dump(pos, handle)
    return pos

def __get_infpower_split() -> nx.Graph:
    file = PROJECT_ROOT / 'infpower' / 'split.pickle'
    if file.exists():
        with open(file, 'rb') as handle:
            return pickle.load(handle)
    G = infpower['inf-power']
    lcvs = __get_pickled_local_cutvertices(G)
    split_at_local_cutvertices(G, lcvs, inplace=True)
    with open(file, 'wb') as handle:
        pickle.dump(G, handle)
    return G

def __read_adjacency_mat_list(n: int, total: int, file: Path) -> List[np.ndarray]:
    mats = [np.zeros((n, n)) for _ in range(total)]
    read = 0
    with open(file, 'r') as f:
        while read != total:
            for i in range(n):
                mats[read][i,:] = list(map(int, f.readline().strip().split()))
            # Discard blank line
            _ = f.readline()
            read += 1
    return mats

def __ramt_boilerplate(n: int, total: int) -> List[nx.Graph]:
    return list(
        map(
            nx.from_numpy_matrix,
            __read_adjacency_mat_list(n, total, PROJECT_ROOT / f'list_{total}_graphs.mat')
        )
    )

def __get_connected_graphs_3_vertices() -> List[nx.Graph]:
    return __ramt_boilerplate(3, 2)

def __get_connected_graphs_4_vertices() -> List[nx.Graph]:
    return __ramt_boilerplate(4, 6)

def __get_connected_graphs_5_vertices() -> List[nx.Graph]:
    return __ramt_boilerplate(5, 21)

def CPL(G: nx.Graph) -> float:
    '''
        Function that returns the characteristic path length of a graph.

        Parameters
        ----------
        G: nx.Graph
        
        Notes
        -----
        see my thesis for definition

        Returns
        -------
        float
    '''
    # Obtain the number of vertices.
    V: int = G.number_of_nodes()
    # Obtain all shortest path lengths between reachable vertices.
    SPL: Dict[Vertex, Dict[Vertex, int]] = dict(nx.all_pairs_shortest_path_length(G))
    # Obtain all possible combinations of pairs of vertices.
    combos = itertools.combinations(G.nodes(), 2)
    # Calculate the total sum.
    L: int = sum(SPL[u].get(v, 0) for u,v in combos)
    # Return characteristic path length.
    return 2 * L / (V * (V - 1))

def CC(G: nx.Graph) -> float:
    '''
        Function that calculates the clustering coefficient of a graph.

        Parameters
        ----------
        G: nx.Graph

        Notes
        -----
        See thesis, thanks.

        Returns
        -------
        float
    '''
    def LCC(v: Vertex) -> float:
        '''
            Function that calculates the local clustering coefficient for a
            vertex in a graph.

            Parameters
            ----------
            v: Vertex

            Notes
            -----
            See thesis, thanks.

            Returns
            -------
            float
        '''
        # Obtain degree of v in G.
        d: int = G.degree(v)
        # Obtain the neighbours of v in G.
        neighbours: List[Vertex] = list(G.neighbors(v))
        # For each combination of neighbour, see if the edge is present in G.
        numerator: int = sum(
            G.has_edge(x,y) for x,y in itertools.combinations(neighbours, 2)
        )
        # Return the local clustering coefficient.
        return 2 * numerator / (d * (d - 1))
    # Obtain all vertices of degree at least 2.
    verts: List[Vertex] = list(
        filter(
            lambda node: G.degree(node) >= 2,
            G.nodes()
        )
    )
    # Calculate all local clustering coefficients.
    local_clustering_coeffs: List[float] = list(map(LCC, verts))
    # Return the average.
    return sum(local_clustering_coeffs) / len(verts)

# TESTING THE DRAWING FUNCTIONS IN visualise.py

def try_draw_graph():
    G: nx.Graph = NDMJS20['net_green_eggs']
    layout: callable = nx.kamada_kawai_layout
    pos: Dict[Vertex, Point2d] = layout(G)
    method: str = 'min'
    draw_graph(G, pos, method=method)

def try_draw_local_cutvertices():
    MIN_RADIUS: int = 4
    G: nx.Graph = NDMJS20['net_coli']
    with open(_pickle_name(G.name), 'rb') as handle:
        local_cutvertices: Dict[Vertex, int] = pickle.load(handle)
    local_cutvertices: Dict[Vertex, int] = {
        k:v for k,v in local_cutvertices.items() if v >= MIN_RADIUS
    }
    layout: callable = nx.kamada_kawai_layout
    pos: Dict[Vertex, Point2d] = layout(G)
    draw_local_cutvertices(G, pos, local_cutvertices, overwrite=True)

def try_draw_split_vertices(graph: str, min_locality: int=None):
    import pickle
    G: nx.Graph = NDMJS20[graph]
    min_locality: int = min_locality or 0
    with open(_pickle_name(G.name), 'rb') as handle:
        local_cutvertices: List[LocalCutvertex] = [
            lcv for lcv in pickle.load(handle)
            if lcv.locality >= min_locality
        ]
    layout: callable = nx.kamada_kawai_layout
    draw_split_vertices(G, layout, local_cutvertices)

def try_draw_locality_heatmap():
    G: nx.Graph = NDMJS20['net_AG']
    with open(_pickle_name(G.name), 'rb') as handle:
        local_cutvertices: Dict[Vertex, int] = pickle.load(handle)
    layout: callable = nx.kamada_kawai_layout
    draw_locality_heatmap(G, layout, local_cutvertices)

# VISUALISING THINGS

def show_local_cutvertex(G: nx.Graph, v: LocalCutvertex):
    '''
        Makes three plots: first is G with v highlighted
    '''
    pass

# DATASET PROCESSING BATCH FUNCTIONS

def draw_local_cutvertices_Network_Data_MJS20(min_radius: int=None, overwrite: bool=False):
    if overwrite:
        print('OVERWRITING IMAGES OF LOCAL CUTVERTICES!')
        print('Confirm? [y/n]')
        while True:
            print('>>> ', end='')
            ans: str = input()
            if ans.lower().startswith('y'):
                print('Overwrite confirmed.')
                break
            elif ans.lower().startswith('n'):
                print('Overwrite aborted!')
                exit()
    layout: callable = nx.kamada_kawai_layout
    if min_radius is None:
        min_radius: int = 0
    categories: List[str] = 'FoodWebs Genetic Language Metabolic Neural Social Trade'.split()
    datasets: List[List[nx.Graph]] = list(map(lambda x: NDMJS20[x], categories))
    total_start: float = time.perf_counter()
    for i, (category, dataset) in enumerate(zip(categories, datasets)):
        print(f'[{i+1}/{len(categories)}] {category}')
        G: nx.Graph
        for i, G in enumerate(dataset):
            with open(_pickle_name(G.name), 'rb') as handle:
                local_cutvertices: Dict[Vertex, int] = pickle.load(handle)
            local_cutvertices: Dict[Vertex, int] = {
                k:v for k,v in local_cutvertices.items()
                if v >= min_radius
            }
            if not local_cutvertices:
                print(f'\t({i+1}/{len(dataset)}) Nothing to draw for {G.name.name}...')
                continue
            pos = layout(G)
            print(f'\t({i+1}/{len(dataset)}) Drawing {pluralise(len(local_cutvertices), "local cutvertex")} for {G.name.name}...')
            start = time.perf_counter()
            draw_local_cutvertices(G, pos, local_cutvertices, overwrite=overwrite)
            end = time.perf_counter()
            print('\tTook', sec2str(end-start, rounding=3))
    total_end: float = time.perf_counter()
    dur: float = total_end - total_start
    print('Total drawing time:', sec2str(dur, rounding=2))

def NDMJS20_sorted_by_local_cutvertex_count():
    graphs = np.array(
        __get_Network_Data_MJS20_graphs(), dtype=object
    )
    local_cutvertices: List[List[LocalCutvertex]] = []
    for graph in graphs:
        with open(_pickle_name(graph.name), 'rb') as handle:
            local_cutvertices.append(pickle.load(handle))
    min_locality: int = 4
    scores = np.array(
        [
            sum(
                v for k,v in Counter(lcv.locality for lcv in lc).items()
                if k > min_locality
            )
            for lc in local_cutvertices
        ]
    )
    inds = np.argsort(scores)[::-1] # descending order
    top_n: int = len(inds)
    with open('sorted_by_local_cutvertex_count.txt', 'w') as f:
        for i, graph in enumerate(graphs[inds][:top_n]):
            idx = inds[i]
            lc: List[LocalCutvertex] = local_cutvertices[idx]
            rad_count = Counter(filter(lambda x: x > min_locality, [lcv.locality for lcv in lc]))
            f.write(f'Position #{i+1} (Score {scores[idx]}): {graph.name.stem}\n')
            for rad, count in rad_count.items():
                f.write(f">>> {pluralise(count, f'{rad}-local cutvertex')}\n")
            f.write('-'*20 + '\n')
    print('Done.')

def graphs_with_no_neighbouring_local_cutvertices():
    min_locality: int = 4
    max_locality: int = 10
    top: int = 17
    graph_names: List[str] = []
    local_cutvertices: List[List[LocalCutvertex]] = []
    print('Reading in graphs...')
    with open('sorted_by_local_cutvertex_count.txt', 'r') as f:
        while top:
            graph_name: str = f.readline().strip().split(':')[1].strip()
            graph_names.append(graph_name)
            graph: nx.Graph = NDMJS20[graph_name]
            with open(_pickle_name(graph.name), 'rb') as handle:
                local_cutvertices.append(pickle.load(handle))
            line: str = f.readline()
            while not all(x == '-' for x in line.strip()):
                line: str = f.readline()
            top -= 1
    print(len(graph_names), 'graphs read, determining non-problematic ones...')
    non_problematic: List[Tuple[int, str]] = []
    lcv: List[LocalCutvertex]
    for j, (graph_name, lcv) in enumerate(zip(graph_names, local_cutvertices)):
        print('Processing graph number', j+1)
        locality: int = min_locality
        while True:
            # Obtain all the local cutvertices.
            all_lcvs: List[Vertex] = set(
                lc.vertex for lc in lcv
                if lc.locality == locality
            )
            if not all_lcvs:
                if locality >= max_locality:
                    break
                locality += 1
                continue
            # For each local cutvertex, see if all of its edge-partition
            # subsets don't contain another local cutvertex.
            passed_all: bool = True
            for lc in lcv:
                passed: bool = all(
                    other_lc not in subset
                    for subset in lc.edge_partition
                    for other_lc in lcv if other_lc != lc
                )
                if not passed:
                    passed_all: bool = False
                    break
            if passed_all:
                non_problematic.append((locality, graph_name))
            locality += 1
    if non_problematic:
        for locality, name in non_problematic:
            print(name, 'is not problematic at locality', locality)
    else:
        print('abandon ship, include the stats graph and write the report')

# FIGURES

def __radii_Network_Data_MJS20(min_locality: int=None):
    from . import _pickle_name
    import pickle
    # Obtain all the graphs in the dataset.
    graphs: List[nx.Graph] = __get_Network_Data_MJS20_graphs()
    # Concatenate all the radii.
    min_locality: int = min_locality or 0
    radii = []
    for graph in graphs:
        with open(_pickle_name(graph.name), 'rb') as handle:
            local_cutvertices: List[LocalCutvertex] = pickle.load(handle)
        radii.extend(
            [
                lcv.locality for lcv in local_cutvertices
                if lcv.locality >= min_locality
            ]
        )
    radii = Counter(radii)
    keys, vals = radii.keys(), radii.values()
    fig = plt.gcf()
    fig.set_size_inches(5, 3)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_yticks(list(vals))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.bar(keys, vals)
    threshold_line_params = {
        'linewidth': 1,
        'color': 'black',
        'linestyle': 'dashed',
        'alpha': 0.25
    }
    for val in vals:
        ax.axhline(y=val, **threshold_line_params)
    ax.set_xlabel('local cutvertex radius $r$')
    ax.set_ylabel('number of $r$-local cutvertices')
    ax.set_title(
        'Distribution of local cutvertex radii in the '
        + escape_underscore('Network_Data_MJS20')
        + f' dataset ({len(graphs)} graphs)'
    )
    plt.tight_layout()
    plt.show()

def __number_of_local_cutvertices_Network_Data_MJS20():
    raise NotImplementedError('it is implemented, just worthless')
    from . import _pickle_name
    import pickle
    # Obtain all the graphs in the dataset.
    graphs: List[nx.Graph] = __get_Network_Data_MJS20_graphs()
    # Obtain the number of local cutvertices in each graph.
    nums: List[int] = []
    for graph in graphs:
        with open(_pickle_name(graph.name), 'rb') as handle:
            lcvs: List[LocalCutvertex] = pickle.load(handle)
        nums.append(len(lcvs))
    # Get the counts.
    nums = Counter(nums)
    keys, vals = nums.keys(), nums.values()
    fig = plt.gcf()
    fig.set_size_inches(5, 3)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xticks(list(vals))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.bar(keys, vals)
    # threshold_line_params = {
    #     'linewidth': 1,
    #     'color': 'black',
    #     'linestyle': 'dashed',
    #     'alpha': 0.25
    # }
    # for val in vals:
    #     ax.axhline(y=val, **threshold_line_params)
    # ax.set_xlabel('local cutvertex radius $r$')
    # ax.set_ylabel('number of $r$-local cutvertices')
    ax.set_title(
        'Distribution of the number of local cutvertices in the '
        + escape_underscore('Network_Data_MJS20')
        + f' dataset ({len(graphs)} graphs)'
    )
    plt.tight_layout()
    plt.show()

def _interim_report_1_figure():
    '''
            o-------o
        / |       | \
        o---o-------o---o
        |   | \ V / |   |
        |   |   o   |   |
        |   | /   \ |   |
        o---o-------o---o
        \ |       | /
            o-------o   
    '''
    local_cutvertex = 'v'
    G: nx.Graph = nx.cycle_graph(8)
    nx.add_cycle(G, range(8, 12))
    G.add_edges_from([
        (6,8), (7,8),
        (0,9), (1,9),
        (2,10), (3,10),
        (4,11), (5,11),
    ])
    G.add_edges_from([(i, local_cutvertex) for i in range(8, 12)])
    pos = {
        0: (0,3),
        1: (1,4),
        2: (3,4),
        3: (4,3),
        4: (4,1),
        5: (3,0),
        6: (1,0),
        7: (0,1),
        8: (1,1),
        9: (1,3),
        10: (3,3),
        11: (3,1),
        local_cutvertex: (2,2)
    }
    ball_edges = set((i, local_cutvertex) for i in range(8, 12))
    ball_nodes = set(range(8, 12))
    ball_node_size = 150
    ball_colour = 'tab:pink'
    legend_loc = 'lower right'
    other_node_size = 100
    plt.subplot(121)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes() - ball_nodes - {local_cutvertex,}, label=f'$u\in V\setminus V(B_1({local_cutvertex}))$', node_size=other_node_size)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges() - ball_edges, alpha=0.4, label=f'$xy\in E\setminus E(B_1({local_cutvertex}))$')
    nx.draw_networkx_nodes(G, pos, nodelist=ball_nodes.union({local_cutvertex,}), label=rf'$V(B_1({local_cutvertex}))$', node_color=ball_colour, node_size=ball_node_size)
    nx.draw_networkx_labels(G, pos, labels={local_cutvertex: f'${local_cutvertex}$'})
    nx.draw_networkx_edges(G, pos, edgelist=ball_edges, edge_color=ball_colour, width=1.5, label=rf'$E(B_1({local_cutvertex}))$')
    plt.legend(scatterpoints=1, loc=legend_loc)
    plt.title(f'Graph $G$ with $2$-local cutvertex ${local_cutvertex}$')
    plt.subplot(122)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes() - ball_nodes - {local_cutvertex,}, label=fr'$u\in V\setminus\left\lbrace {local_cutvertex}\right\rbrace$', node_size=other_node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=ball_nodes, label=rf'$V(B_1({local_cutvertex}))\setminus\left\lbrace {local_cutvertex}\right\rbrace$', node_color=ball_colour, node_size=ball_node_size)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges() - ball_edges, alpha=0.4, label=fr'$xy\in \left\lbrace xy\in E\mid x\neq {local_cutvertex}, y\neq {local_cutvertex}\right\rbrace$')
    plt.legend(scatterpoints=1, loc=legend_loc)
    plt.title(f'Graph $G$ with $2$-local cutvertex ${local_cutvertex}$ removed')
    plt.show()

def w13_every_other_pair_of_spokes_removed():
    node_size = 500
    font_size = 16
    width = 2
    hub = 0
    G: nx.Graph = __get_W13_special(hub=hub)
    matching_offset: float = 0.
    global_offset: float = 0.
    pos = {i+hub+1: point for i,point in enumerate(polygon(12, 2, rotate_degrees=matching_offset + global_offset))}
    pos[hub] = (0,0)
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    nx.draw_networkx(G, pos, node_size=node_size, font_size=font_size, width=width, with_labels=True, font_color='w', ax=ax)
    ax.set_title('Graph $H$, isomorphic to $W^{12}$ with every other pair of spokes removed')
    ax = axes[1]
    B = ball(G, hub, 1.5)
    nx.draw_networkx(B, pos, node_size=node_size, font_size=font_size, width=width, with_labels=True, font_color='w', ax=ax)
    ax.set_title(f'Graph $H^\prime$, a copy of $B_H({hub},\\frac{{3}}{{2}})$')
    plt.show()

def definition_splitting_local_cutvertex(fname: Path=None, dpi: int=600):
    # Obtain the graph.
    hub: int = 1
    G: nx.Graph = __get_W13_special(hub=hub)
    # Create the labels (all empty strings except for hub).
    hub_name: str = 'v'
    labels: Dict[int, str] = {
        v: f'${hub_name}$' * (v == hub) for v in G.nodes()
    }
    # Construct the figure and axes.
    fig, axes = plt.subplots(1, 3)
    if fname is not None:
        fig.set_size_inches(9, 4)
    # Introduce some figure hyperparameters.
    NODE_SIZE: int = 250
    DEFAULT_EDGE_WIDTH: float = 2
    THICK_EDGE_WIDTH: float = 3
    FONT_WEIGHT: str = 'heavy'
    EDGECOLORS: str = 'k'
    
    ### Construct the different graphs we'll be plotting.
    ## G is the graph on the left, plain and simple, already there.
    ## H is the graph in the middle, apart from some highlighting
    ## when being drawn, it is identical to G.
    H: nx.Graph = G
    ## I is the graph on the right, it'll be G split at hub with radius r.
    r: int = 4
    ball_around_hub: nx.Graph = ball(G, hub, r/2)
    punctured: nx.Graph = nx.subgraph_view(ball_around_hub, filter_node=lambda x: x != hub)
    punctured_components: List[Set[Vertex]] = list(nx.connected_components(punctured))
    neighbourhood: Set[Vertex] = set(G.neighbors(hub))
    edge_partition: Set[Tuple[Vertex, ...]] = set(
        tuple(neighbourhood.intersection(comp)) for comp in punctured_components
    )
    local_cutvertex: LocalCutvertex = LocalCutvertex(
        vertex=hub, locality=r, edge_partition=edge_partition
    )
    I: nx.Graph = split_at_local_cutvertices(G, [local_cutvertex])
    split_components: List[Set[Vertex]] = list(
        nx.connected_components(
            nx.subgraph_view(I, filter_node=lambda node: node != hub)
        )
    )
    # Add the split vertices to the labels dictionary.
    for node, split in I.nodes(data='split'):
        if split:
            labels[node] = ''
    # Obtain the required amount of visually distinct colours.
    num_colours: int = len(punctured_components) + 2
    COMPONENT_COLOURS: List[str] = [f'xkcd:{yuh}' for yuh in 'teal,barney purple,salmon'.split(',')]
    DEFAULT_COLOUR: str = 'xkcd:sky blue'
    BALL_COLOUR: str = 'xkcd:pinkish red'
    # Obtain the layout.
    matching_offset: float = 15.
    global_offset: float = 210.
    pos = {i+2: point for i,point in enumerate(polygon(12, 2, rotate_degrees=matching_offset + global_offset))}
    pos[hub] = (0,0)
    triangle = list(polygon(3, 1, rotate_degrees=global_offset))
    triangle[0], triangle[1] = triangle[1], triangle[0]
    pos.update({
        node: triangle.pop()
        for node, split in I.nodes(data='split') if split
    })
    ## Proceed to draw.
    # Draw G, plain and simple, with the hub highlighted.
    ax = axes[0]
    ax.set_title(f'Graph $G$ with a ${r}$-local cutvertex ${hub_name}$')
    ax.set_aspect('equal')
    nx.draw_networkx(
        G, pos,
        node_color=DEFAULT_COLOUR,
        edge_color=DEFAULT_COLOUR,
        node_size=NODE_SIZE,
        edgecolors=EDGECOLORS,
        width=DEFAULT_EDGE_WIDTH,
        labels=labels,
        font_weight=FONT_WEIGHT,
        ax=ax,
    )
    # Draw H, which is G with the ball highlighted.
    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_title(rf'$G$ with $B\left({hub_name}, \frac{{{r}}}{{{2}}}\right)$ highlighted')
    non_ball_nodes: Set[Vertex] = G.nodes() - ball_around_hub.nodes()
    non_ball_edges: List[Tuple[Vertex, Vertex]] = G.edges() - ball_around_hub.edges()
    nx.draw_networkx(
        ball_around_hub, pos,
        node_size=NODE_SIZE,
        node_color=BALL_COLOUR,
        width=THICK_EDGE_WIDTH,
        edgecolors=EDGECOLORS,
        edge_color=BALL_COLOUR,
        labels=labels,
        font_weight=FONT_WEIGHT,
        ax=ax
    )
    nx.draw_networkx(
        G, pos,
        node_size=NODE_SIZE,
        nodelist=non_ball_nodes,
        node_color=DEFAULT_COLOUR,
        edgecolors=EDGECOLORS,
        edgelist=non_ball_edges,
        width=DEFAULT_EDGE_WIDTH,
        edge_color=DEFAULT_COLOUR,
        labels=labels,
        font_weight=FONT_WEIGHT,
        ax=ax
    )
    # Draw I, which is G split at v.
    ax = axes[2]
    ax.set_aspect('equal')
    ax.set_title(f'$G$ split at ${hub_name}$, with square split vertices')
    # Draw the non-ball nodes and edges.
    nx.draw_networkx(
        I, pos,
        nodelist=non_ball_nodes.union({hub}),
        node_size=NODE_SIZE,
        node_color=DEFAULT_COLOUR,
        edgecolors=EDGECOLORS,
        edgelist=non_ball_edges,
        width=DEFAULT_EDGE_WIDTH,
        edge_color=DEFAULT_COLOUR,
        labels=labels,
        font_weight=FONT_WEIGHT,
        ax=ax
    )
    # Draw each component of the punctured ball in I, and its split vertex.
    split_vertices: List[Vertex] = [
        next(
            vertex for vertex, split in I.nodes(data='split')
            if split and any(
                I.has_edge(vertex, c_i) for c_i in component
            )
        ) for component in punctured_components
    ]
    edgelists: List[List[Tuple[Vertex, Vertex]]] = [
        list(nx.subgraph_view(I, filter_node=lambda x: x in component).edges())
        + [(split_vertex, n) for n in I.neighbors(split_vertex) if n != hub]
        for split_vertex, component in zip(split_vertices, punctured_components)
    ]
    for component, split_vertex, edgelist, colour in zip(punctured_components, split_vertices, edgelists, COMPONENT_COLOURS):
        nx.draw_networkx(
            I, pos,
            nodelist=component,
            node_size=NODE_SIZE,
            node_color=colour,
            edgecolors=EDGECOLORS,
            edgelist=edgelist,
            width=THICK_EDGE_WIDTH,
            edge_color=colour,
            with_labels=False,
            ax=ax
        )
        nx.draw_networkx_nodes(
            I, pos,
            nodelist=[split_vertex],
            node_size=NODE_SIZE,
            edgecolors=EDGECOLORS,
            node_shape='s',
            node_color=colour,
            ax=ax
        )
        nx.draw_networkx_edges(
            I, pos,
            edgelist=I.edges(hub),
            width=DEFAULT_EDGE_WIDTH,
            edge_color=DEFAULT_COLOUR,
            ax=ax
        )
    # Save fig.
    if fname is None:
        plt.show()
    else:
        if not isinstance(fname, str):
            fname: str = path_to_str(fname)
        plt.savefig(fname, dpi=dpi)

def CNCL_MORN():
    G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    connected_components: List[Set[Vertex]] = list(nx.connected_components(G))
    closest_non_component_leaf: Dict[Vertex, Tuple[Vertex, float]] = closest_non_component_leaf_MORN(G)
    items = list(closest_non_component_leaf.items())
    random.shuffle(items)
    for leaf, (closest_leaf, distance) in items:
        leaf_component: Set[Vertex] = next(comp for comp in connected_components if leaf in comp)
        closest_leaf_component: Set[Vertex] = next(
            comp for comp in connected_components if closest_leaf in comp
        )
        fig, ax = plt.subplots()
        ax.set_title(f'Distance: ${distance}$')
        excluded: Set[Vertex] = G.nodes() - leaf_component - closest_leaf_component
        
        nx.draw_networkx_nodes(G, pos, nodelist=excluded, node_size=1, alpha=0.2)
        nx.draw_networkx_edges(G, pos, edgelist=[(x,y) for x,y in G.edges() if x in excluded and y in excluded], alpha=0.1)
        
        nx.draw_networkx_nodes(G, pos, nodelist=leaf_component - {leaf}, node_size=5, alpha=0.4, node_color='g')
        nx.draw_networkx_edges(G, pos, edgelist=[(x,y) for x,y in G.edges() if x in leaf_component and y in leaf_component], alpha=0.4, edge_color='g')
        
        nx.draw_networkx_nodes(G, pos, nodelist=closest_leaf_component - {closest_leaf}, node_size=5, alpha=0.4, node_color='purple')
        nx.draw_networkx_edges(G, pos, edgelist=[(x,y) for x,y in G.edges() if x in closest_leaf_component and y in closest_leaf_component], alpha=0.4, edge_color='purple')
        
        nx.draw_networkx_nodes(G, pos, nodelist=[leaf], node_size=10, node_color='r')
        nx.draw_networkx_nodes(G, pos, nodelist=[closest_leaf], node_size=10, node_color='hotpink')

        plt.show()

def CNCV_MORN(fname: Union[str, Path]=None):
    raise NotImplementedError("not really, just haven't fixed this")
    # TODO: BASICALLY FIX THIS
    G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    fig, ax = plt.subplots()
    closest_non_component_vertices: Dict[Vertex, Tuple[Vertex, float]] = closest_non_component_vertex_MORN(G)
    distances: List[float] = list(map(operator.itemgetter(1), closest_non_component_vertices.values()))
    zeros: int = distances.count(0)
    bins = [0, 1, 10, 100, 1_000]
    ax.set_xscale('log')
    ax.set_xticks(bins)
    ax.set_xlabel('distance in meters')
    # ax.set_yscale('log')
    ax.set_ylabel('count')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_title('Distribution of distances to closest non-component vertex')
    start: float = time.perf_counter()
    counts, edges, bars = ax.hist(distances, bins=bins, align='right')
    end: float = time.perf_counter()
    ax.bar_label(bars)
    if fname is None:
        plt.show()
    else:
        if not isinstance(fname, str):
            fname: str = path_to_str(fname)
        plt.savefig(fname)

def CNCV_MORN_table():
    G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    fig, ax = plt.subplots()
    closest_non_component_vertices: Dict[Vertex, Tuple[Vertex, float]] = closest_non_component_vertex_MORN(G)
    distances: List[float] = list(map(operator.itemgetter(1), closest_non_component_vertices.values()))
    last: int = 2_000
    my_bins = [0, 1, 10, 100, 1_000, last]
    my_bins = [
        (x,y) for x,y in zip(my_bins[:-1], my_bins[1:])
    ]
    my_bins.insert(0, 0)
    my_bins.append(last)
    # my_bins = [0, (a,b), (c,d), ..., (z,last), last]
    my_bins_labels = []
    counts = []
    for i, get_in_the in enumerate(my_bins):
        # Special case for the last bin.
        if i == len(my_bins) - 1:
            counts.append(
                sum(distance > get_in_the for distance in distances)
            )
            my_bins_labels.append(f'$> {get_in_the}$')
            break
        # Not dealing with the last bin.
        try:
            x, y = get_in_the
            counts.append(
                sum(
                    x < distance <= y for distance in distances
                )
            )
            my_bins_labels.append(rf'$\left( {x},{y}\right\rbrack$')
        except TypeError:
            counts.append(distances.count(get_in_the))
            my_bins_labels.append(f'${get_in_the}$')

    assert len(my_bins) == len(counts), 'wesh gros'
    print('Distance (meters) & ', end='')
    for i, label in enumerate(my_bins_labels):
        print(label, end='')
        if i != len(my_bins_labels) - 1:
            print(' & ', end='')
        else:
            print(r'\\')
    print('Count & ', end='')
    for i, count in enumerate(counts):
        print(f'${count}$', end='')
        if i != len(counts) - 1:
            print(' & ', end='')
        else:
            print(r'\\')

def OG_SIZE_MORN(fname: Union[str, Path]=None):
    '''
        Visualise number of and size of overlapping groups of vertices
        in the MORN dataset.
    '''
    # Obtain the dataset.
    G: nx.Graph = __get_MORN()
    # Obtain the overlapping groups.
    groups: List[Set[Vertex]] = overlapping_groups_MORN(G)
    # Obtain their sizes.
    sizes: List[int] = list(map(len, groups))
    overlapping_points: int = sum(sizes)
    num_nodes: int = G.number_of_nodes()
    non_overlapping_points: int = num_nodes - overlapping_points
    # Create Counter from sizes.
    c = Counter(sizes)
    c[1] = non_overlapping_points
    # Add non-overlapping points.
    # Plot sizes.
    fig, ax = plt.subplots()
    keys, vals = c.keys(), c.values()
    fig.set_size_inches(6, 3.5)
    ax.set_yscale('log')
    ax.set_yticks(list(vals))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.bar(keys, vals)
    threshold_line_params = {
        'linewidth': 1,
        'color': 'black',
        'linestyle': 'dashed',
        'alpha': 0.25
    }
    for val in vals:
        ax.axhline(y=val, **threshold_line_params)
    ax.set_xlabel('size of group of overlapping vertices')
    ax.set_ylabel('number of overlapping groups')
    ax.set_title('Sizes of overlapping groups of vertices in the MRN dataset')
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        if not isinstance(fname, str):
            fname: str = path_to_str(fname)
        plt.savefig(fname, dpi=300)

def OG_COLLAPSING_NUM_COMPONENTS_MORN(fname: Union[str, Path]=None):
    # Get graph.
    G: nx.Graph = __get_MORN()
    # Get number of connected components.
    component_count_accumulator: List[int] = G.graph['component_count_accumulator']
    # Plot that shit
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 3.5)
    # ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    threshold_line_params = {
        'linewidth': 1,
        'color': 'black',
        'linestyle': 'dashed',
        'alpha': 0.25
    }
    xticks = list(range(0, 50_001, 10_000))
    xticks.append(len(component_count_accumulator))
    vals = [0, -1]
    xvals = [xticks[i] for i in vals]
    yvals = [component_count_accumulator[i] for i in vals]
    for xval, yval in zip(xvals, yvals):
        ax.axvline(x=xval, **threshold_line_params)
        ax.axhline(y=yval, **threshold_line_params)
    ax.set_xticks(xticks)
    ax.set_yticks(yvals)
    ax.plot(
        range(len(component_count_accumulator)),
        component_count_accumulator,
        linestyle='--',
        color='xkcd:red'
    )
    ax.set_xlabel('overlapping groups collapsed')
    ax.set_ylabel('number of components')
    # ax.set_title('Number of components in $G$ while collapsing')
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        if not isinstance(fname, str):
            fname: str = path_to_str(fname)
        plt.savefig(fname, dpi=300)

def REDUNDANT_MORN(fname: Union[str, Path]=None, figure_size: FigureSize=A4, dpi: int=300):
    '''
        Plot the MORN dataset with redundant vertices highlighted.
    '''
    # Get old savings ratio.
    G: nx.Graph = __get_MORN()
    old_savings_ratio: float = G.graph['savings_ratio']
    del G
    # Get the flattened version of the MORN dataset.
    flatten: Path = PROJECT_ROOT / 'datasets' / 'MajorOpenRoadNetwork' / 'FlattenGroups.pickle'
    G: nx.Graph
    with open(flatten, 'rb') as handle:
        _, G = pickle.load(handle)
    # Obtain redundant vertices in F.
    redundant: Set[Vertex] = set(redundant_points_MORN(G))
    # Calculate new savings ratio.
    num_nodes: int = G.number_of_nodes()
    new_savings_ratio: float = 1 - (num_nodes - len(redundant)) / num_nodes
    # Compare.
    assert old_savings_ratio == new_savings_ratio, (old_savings_ratio, new_savings_ratio)
    # Plot this shit.
    pos = G.graph['pos']
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(*figure_size)
    fig.set_dpi(dpi)
    # Draw regular vertices in blue and redundant ones in red.
    NODE_SIZE: int = 10
    nx.draw_networkx_nodes(
        G, pos,
        node_size=NODE_SIZE, node_color='blue',
        nodelist=G.nodes() - redundant,
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_size=NODE_SIZE, node_color='red',
        nodelist=redundant,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=dpi)

def REDUNDANT_MORN_stats():
    # Obtain flattened graph.
    flatten: Path = PROJECT_ROOT / 'datasets' / 'MajorOpenRoadNetwork' / 'FlattenGroups.pickle'
    G: nx.Graph
    with open(flatten, 'rb') as handle:
        _, G = pickle.load(handle)
    # Obtain redundant vertices.
    redundant: List[Vertex] = redundant_points_MORN(G)
    # Compute savings ratio.
    num_nodes: int = G.number_of_nodes()
    savings_ratio: float = 1 - (num_nodes - len(redundant)) / num_nodes
    # Print stats.
    print('Total vertices:', num_nodes)
    print('Redundant vertices:', len(redundant))
    print('Savings ratio:', round(savings_ratio*100, 3))

def OG_MORN(fname: Union[str, Path]=None, group_size: int=None, seed: int=None):
    '''
        Picks an overlapping group of vertices and visualises their components
        in distinct colours.

        Parameters
        ----------
        fname: Union[str, Path], optional
        group_size: int, optional
            If specified, the size of the random group to pick, random otherwise.
        seed: int, optional
            If specified, the random seed to initialise the state before any random
            operation.
    '''
    # Obtain the dataset.
    G: nx.Graph = __get_MORN()
    # Obtain the overlapping groups.
    groups: List[Set[Vertex]] = overlapping_groups_MORN(G)
    # Obtain their sizes.
    sizes: List[int] = list(map(len, groups))
    # Check if a seed has been specified.
    if seed is not None:
        random.seed(seed)
    # Check if a group size has been specified.
    if group_size is not None:
        valid_groups: List[int] = [i for i,g in enumerate(groups) if len(g) == group_size]
        index: int = random.choice(valid_groups)
    else:
        index: int = random.choice(range(len(groups)))
    # Get the group.
    group: Set[Vertex] = groups[index]
    # Get the required number of distinct colours.
    colours: List[str] = visually_distinct_colours(len(group))
    # Get the Cartesian coordinates.
    pos = G.graph['pos']
    # Get the matplotlib subplot.
    fig, ax = plt.subplots()
    # For each vertex in the group, plot its component in its own colour.
    print(group)
    for vertex, colour in zip(group, colours):
        # Find the component.
        component_vertices: Set[Vertex] = nx.node_connected_component(G, vertex)
        component: nx.Graph = nx.subgraph_view(
            G, filter_node=lambda node: node in component_vertices
        )
        # Plot the component.
        nx.draw_networkx(component, pos, node_color=colour, edge_color=colour, ax=ax)
    # Show the end result.
    if fname is None:
        plt.show()
    else:
        if not isinstance(fname, str):
            fname: str = path_to_str(fname)
        plt.savefig(fname)

def NDMJS20_NO_LOCAL_CUTVX_table(decimal_places: int=4):
    graphs: List[nx.Graph] = __get_NDMJS20_no_local_cutvx()
    columns: List[str] = [
        r'G=\left(V,E\right)',
        r'\lvert V\rvert',
        r'\lvert E\rvert',
        r'\delta\left(G\right)',
        r'\Delta\left(G\right)',
        r'\lvert\mathcal{C}_G\rvert',
        r'L\left(G\right)',
        r'C\left(G\right)',
        r'\rho\left(G\right)'
    ]
    funcs: List[callable] = [
        lambda graph: graph.name.stem,
        nx.number_of_nodes,
        nx.number_of_edges,
        lambda graph: min(map(graph.degree, graph.nodes()), default=0),
        lambda graph: max(map(graph.degree, graph.nodes()), default=0),
        nx.number_connected_components,
        CPL,
        CC,
        nx.density
    ]
    # Print top
    print(r'\hline')
    print(' & '.join(map(lambda s: f'${s}$', columns)) + r'\\')
    # Print values for each graph.
    for graph in graphs:
        values = list(func(graph) for func in funcs)
        for i,value in enumerate(values):
            if not isinstance(value, float):
                continue
            if math.modf(value)[0]:
                values[i] = round(value, decimal_places)
        values = [
            escape_underscore(value) if isinstance(value, str) else f'${value}$'
            for value in values
        ]
        print(r'\hline ' + ' & '.join(values) + r'\\')
    print(r'\hline')

def NDMJS20_NO_LOCAL_CUTVX_LEAVES(layout: callable=None, method: str='min'):
    if layout is None:
        layout: callable = nx.kamada_kawai_layout
    graphs: List[nx.Graph] = __get_NDMJS20_no_local_cutvx()
    keep: Set[str] = {'benguela', 'rat_brain_1', 'rhesus_brain_2'}
    graphs: List[nx.Graph] = list(filter(lambda G: G.name.stem in keep, graphs))
    fig, axes = plt.subplots(1, 3)
    axes = axes.reshape(-1)
    edge_alphas = [0.4, 0.01, 0.5]
    edge_widths = [1, 0.5, 1]
    node_alphas = [0.3, 0.1, 0.3]
    leaf_size: int = None
    for G, ax, edge_alpha, edge_width, node_alpha in zip(graphs, axes, edge_alphas, edge_widths, node_alphas):
        ax.set_facecolor('xkcd:very light pink')
        pos = layout(G)
        bbox = bounding_box_2d(list(pos.values()), fudge=.1)
        (min_x, max_y), (max_x, min_y) = bbox.top_left, bbox.bottom_right
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))
        closest: Dict[Vertex, Vertex] = nearest(pos)
        nodelist, node_size = calculate_marker_sizes(
            ax, pos, 'o', .4, method=method
        )
        # Leaves
        leaves_idx = [i for i,v in enumerate(nodelist) if G.degree(v) == 1]
        leaves = [nodelist[i] for i in leaves_idx]
        if leaf_size is None:
            leaves_size = [node_size[i] for i in leaves_idx]
            leaf_size = leaves_size[0]
        # Draw leaves
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=leaves,
            node_size=leaf_size,
            node_color='xkcd:fire engine red',
            edgecolors=None,
            ax=ax
        )
        # Draw other nodes.
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[x for i,x in enumerate(nodelist) if i not in leaves_idx],
            node_size=[s for i,s in enumerate(node_size) if i not in leaves_idx],
            node_color='xkcd:deep sky blue',
            alpha=node_alpha,
            ax=ax
        )
        # Draw edges.
        if G.name.stem != 'rat_brain_1':
            nx.draw_networkx_edges(
                G, pos,
                alpha=edge_alpha,
                width=edge_width,
                ax=ax
            )
        else:
            # Draw that one edge a bit thicker.
            edges_without_thick_one = set(G.edges())
            thick_one = next((x,y) for (x,y) in edges_without_thick_one if G.degree(x) == 1 or G.degree(y) == 1)
            edges_without_thick_one.remove(thick_one)
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges_without_thick_one,
                alpha=edge_alpha,
                width=edge_width,
                ax=ax
            )
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[thick_one],
                alpha=0.4,
                width=1,
                ax=ax
            )
        # Give the plot a title.
        ax.set_title(escape_underscore(G.name.stem))
    fig.suptitle('Outlier graphs with leaves highlighted in red')
    plt.show()

def NDMJ20_NO_LOCAL_CUTVX():
    offenders: List[nx.Graph] = __get_NDMJS20_no_local_cutvx()
    layout: callable = nx.kamada_kawai_layout
    method: str = 'min'
    fig, axes = plt.subplots()
    d: Path = PROJECT_ROOT.parent / 'Thesis' / 'ndmjs20'
    d.mkdir(exist_ok=True)
    for G in offenders:
        draw_graph(G, layout(G), method=method, fig_size=(5, 5))
        plt.tight_layout()
        plt.title('')
        plt.savefig(path_to_str(d / f'{G.name.stem}.png'), dpi=300)
        plt.gcf().clear()
    exit()

def NDMJS20_NO_LOCAL_CUTVX_HYPOTHESIS():
    graphs: List[nx.Graph] = [
        G for G in __get_NDMJS20_no_local_cutvx()
        # if not any(G.degree(v) == 1 for v in G.nodes())
    ]
    for G in graphs:
        to_remove: List[Vertex] = [v for v in G.nodes() if G.degree(v) == 1]
        if to_remove:
            G.remove_nodes_from(to_remove)
    assert len(graphs) == 13
    # Check that each neighbourhood isn't disconnected.
    for i,G in enumerate(graphs):
        print(f'Graph #{i+1} {G.name.stem} passes hypothesis? ...', end='\r')
        print('\t'*7, end='')
        for v in G.nodes():
            # Obtain the induced subgraph.
            Nv: Set[Vertex] = set(G.neighbors(v))
            GNv: nx.Graph = nx.subgraph_view(
                G, filter_node=lambda x: x in Nv
            )
            # Check its connectivity.
            if not nx.is_connected(GNv):
                print('No :(')
                break
        else:
            # Lol, finally using a for/else
            print('Yes :)')

def NDMJS20_MOST_K_LOCALS(k: int) -> List[Tuple[str, int]]:
    '''
        Given a locality, sorts the graphs in the NDMJS20 dataset
        by their number of k-local cutvertices.
    '''
    graphs: List[nx.Graph] = __get_Network_Data_MJS20_graphs()
    yuh = [
        (
            sum(1 for lcv in __get_pickled_local_cutvertices(G) if lcv.locality == k),
            G
        )
        for G in graphs
    ]
    yuh.sort(key=operator.itemgetter(0), reverse=True)
    return yuh

def NDMJS20_TRIANGLES_table():
    from scipy.special import comb
    rounding = 2
    fmt = rf'{{:.{rounding}f}}'
    yuh = NDMJS20_MOST_K_LOCALS(3)
    data = []
    for i, (count, G) in enumerate(yuh):
        n_triangles: int = sum(nx.triangles(G).values()) // 3
        t_triangles: int = int(comb(G.number_of_nodes(), 3))
        absent_pcent: float = 100 * n_triangles / t_triangles
        data.append(
            (G.name.stem, count, fmt.format(absent_pcent))
        )
    data = [data[:math.floor(len(data)/2)], data[math.floor(len(data)/2):]]
    for d in data:
        table = tabulate.tabulate(
            d,
            headers=['G', '$n_3$', f't(G) [{rounding}dp]'],
            tablefmt='latex',
        )
        print(table)

def SO_PLOT(layout: callable, names: bool=False):
    # Get the graph.
    G: nx.Graph = stackoverflow['stackoverflow']
    pos = layout(G)
    # Configure the plot.
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    ax.set_facecolor('xkcd:eggshell')
    # Get the groups and colours.
    groups: Set[int] = G.graph['groups']
    colours: List[str] = ['#d8dcd6', '#be6400', '#808000', '#b790d4', '#00ced1', '#ff8c00', '#ffff00', '#00ff00', '#0000ff', '#d8bfd8', '#ff00ff', '#1e90ff', '#ff1493', '#98fb98']
    # Draw the vertices.
    for group, colour in zip(groups, colours):
        nodelist = [node for (node, node_group) in G.nodes(data='group') if node_group == group]
        node_size = [G.nodes[node]['node_size'] / 12 for node in nodelist]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodelist,
            node_size=node_size,
            node_color=colour,
            ax=ax
        )
        if names:
            nx.draw_networkx_labels(
                G, pos,
                labels={
                    node: escape_underscore(node.decode('utf-8')).replace('#', '\#')
                    for node in nodelist
                },
                font_color=font_color,
                font_weight='bold',
                ax=ax
            )
    # Draw the edges.
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.1,
        width=0.5,
        ax=ax
    )
    ax.set_title('stackoverflow Developer Stories tags')
    plt.show()

def SO_PLOT_COMMUNITIES(layout: callable, EDGE_SCALE: float=8.):
    # Get the graph.
    G: nx.Graph = stackoverflow['stackoverflow']
    # Configure the plot.
    # fig, axes = plt.subplots(7, 2)
    # axes = axes.reshape(-1)
    # fig.set_size_inches(10, 10)
    # Get the groups and colours.
    # groups: Set[int] = G.graph['groups']
    colours: List[str] = ['#d8dcd6', '#be6400', '#808000', '#b790d4', '#00ced1', '#ff8c00', '#ffff00', '#00ff00', '#0000ff', '#d8bfd8', '#ff00ff', '#1e90ff', '#ff1493', '#98fb98']
    # Configure the subplots.
    # Draw the vertices.
    subplot_groupings = [
        ((2,), (1,1)),
        ((6,), (1,1)),
        ((8,), (1,1)),
        ((7,9,10,11,12,13,14), (4,2)),
        ((1,3,4,5), (2,2))
    ]
    for groups, subplot_dim in subplot_groupings:
        colour_list = [colours[group-1] for group in groups]
        good = False
        while not good:
            pos = layout(G)
            fig, axes = plt.subplots(*subplot_dim)
            try:
                axes = axes.reshape(-1)
            except AttributeError:
                axes = [axes]
            for group, colour, ax in zip(groups, colour_list, axes):
                nodelist = [node for (node, node_group) in G.nodes(data='group') if node_group == group]
                H: nx.Graph = nx.subgraph_view(G, filter_node=lambda node: node in nodelist)
                node_size = [G.nodes[node]['node_size'] / 12 for node in nodelist]
                ax.set_facecolor('xkcd:eggshell')
                nx.draw_networkx_nodes(
                    H, pos,
                    nodelist=nodelist,
                    node_size=node_size,
                    node_color=colour,
                    ax=ax
                )
                edges = list(H.edges())
                edge_width = [
                    H.edges[edge].get('weight', EDGE_SCALE) / EDGE_SCALE
                    for edge in edges
                ]
                nx.draw_networkx_edges(
                    H, pos,
                    edgelist=edges,
                    width=edge_width,
                    alpha=0.2,
                    edge_color=colour,
                    ax=ax
                )
                font_colour: str = 'k'
                nx.draw_networkx_labels(
                    H, pos,
                    labels={
                        node: lss(node.decode('utf-8'))
                        for node in nodelist
                    },
                    font_color=font_colour,
                    ax=ax
                )
                if len(groups) == 1:
                    ax.set_title(f'Community {group}')
            plt.tight_layout()
            plt.show()
            print('Good [y/n]? >>> ', end='')
            good = input().lower().startswith('y')
            if not good:
                plt.close(fig)

def SO_COMMUNITIES_table():
    G: nx.Graph = stackoverflow['stackoverflow']
    groups: Set[int] = G.graph['groups']
    print(r'\hline')
    _columns: List[str] = ['Community Number', 'Community Vertices']
    columns: List[str] = [rf'\textbf{{{column}}}' for column in _columns]
    print(' & '.join(columns) + r' \\')
    for i, group in enumerate(groups):
        print(r'\hline')
        nodes: List[str] = [
            rf"\textit{{{lss(node.decode('utf-8'))}}}"
            for (node, nodegroup) in G.nodes(data='group')
            if nodegroup == group
        ]
        print(rf'${i+1}$ & {", ".join(nodes)} \\')

def SO_LOCAL_CUTVERTICES_EDGE_PARTITION_GROUPS_table():
    G: nx.Graph = stackoverflow['stackoverflow']
    local_cutvertices: List[LocalCutvertex] = __get_pickled_local_cutvertices(G)
    print(r'\hline')
    columns: List[str] = ['Local Cutvertex $v$', 'Radius $r$', r'$\mathcal{C}\left(B\left(v,\frac{r}{2}\right)-v\right)$']
    print(' & '.join(columns) + r' \\')
    for lcv in local_cutvertices:
        print(r'\hline')
        vertex: str = lss(lcv.vertex.decode('utf-8'))
        radius: str = f'${lcv.locality}$'
        comps: set = {tuple(lss(x.decode('utf-8')) for x in comp) for comp in lcv.edge_partition}
        print(rf'{vertex} & {radius} & ', end='')
        for i,comp in enumerate(comps):
            print(r'$\lbrace$', end='')
            print(', '.join(comp), end='')
            print(r'$\rbrace$', end='')
            if i != len(comps) - 1:
                print(', ', end='')
        print(r' \\')
    print(r'\hline')

def SO_LOCAL_CUTVERTICES_GROUPS_table():
    G: nx.Graph = stackoverflow['stackoverflow']
    local_cutvertices: List[LocalCutvertex] = __get_pickled_local_cutvertices(G)
    print(r'\hline')
    columns: List[str] = [
        '$v$',
        '$r$',
        r'$\lvert\mathcal{C}\rvert$',
        # r'$\mathcal{C}\cap N_G\left(v\right)$',
        r'$\mathcal{C}$'
    ]
    print(' & '.join(columns) + r' \\')
    for lcv in local_cutvertices:
        print(r'\hline')
        vertex: str = lss(lcv.vertex.decode('utf-8'))
        radius: str = f'${lcv.locality}$'
        PB: nx.Graph = nx.subgraph_view(
            ball(G, lcv.vertex, lcv.locality/2), lambda node: node != lcv.vertex
        )
        comps: set = {
            tuple(lss(x.decode('utf-8')) for x in comp)
            for comp in nx.connected_components(PB)
        }
        # nsec: set = {
        #     tuple(lss(x.decode('utf-8')) for x in comp)
        #     for comp in lcv.edge_partition
        # }
        n_comps: int = len(comps)
        print(rf'{vertex} & {radius} & {n_comps} & ', end='')
        # for i,comp in enumerate(nsec):
        #     print(r'$\lbrace$', end='')
        #     print(', '.join(comp), end='')
        #     print(r'$\rbrace$', end='')
        #     if i != len(comps) - 1:
        #         print(', ', end='')
        # print(' & ')
        for i,comp in enumerate(comps):
            print(r'$\lbrace$', end='')
            print(', '.join(comp), end='')
            print(r'$\rbrace$', end='')
            if i != len(comps) - 1:
                print(', ', end='')
        print(r' \\')
    print(r'\hline')

def SO_LOCAL_CUTVERTICES_BALLS(layout: callable, EDGE_SCALE: float=10.):
    G: nx.Graph = stackoverflow['stackoverflow']
    local_cutvertices: List[LocalCutvertex] = __get_pickled_local_cutvertices(G)
    # For each local cutvertex, I want to observe the ball around that local
    # cutvertex, split the ball at that local cutvertex, and observe the different
    # components that arise.
    colours: List[str] = [f'xkcd:{yuh}' for yuh in 'fire engine red,pale blue,chartreuse'.split(',')]
    for lcv in local_cutvertices:
        good: bool = False
        while not good:
            plt.gcf().set_size_inches(6, 6)
            plt.gca().set_facecolor('xkcd:eggshell')
            # Step 1: Obtain the ball.
            B: nx.Graph = ball(G, lcv.vertex, lcv.locality / 2)
            # Step 2: Split at that local cutvertex.
            B_prime: nx.Graph = split_at_local_cutvertices(B, [lcv], inplace=False)
            # Step 3: Obtain the different components.
            comps: List[Set[Vertex]] = list(
                nx.connected_components(
                    nx.subgraph_view(B_prime, filter_node=lambda node: node != lcv.vertex)
                )
            )
            # Step 4: Construct the labels.
            labels: Dict[Vertex, str] = {
                node: lss(node.decode('utf-8')) if not split else ''
                for node, split in B_prime.nodes(data='split')
            }
            # Step 5: Obtain the layout.
            pos = layout(B_prime)
            # Step 6: Obtain the node sizes.
            nodelist = list(B_prime.nodes())
            node_size = [
                B_prime.nodes[node].get('node_size', float('inf')) / 12
                for node in nodelist
            ]
            min_size = min(node_size)
            for i, size in enumerate(node_size):
                if size == float('inf'):
                    node_size[i] = min_size
            # Step 7: Obtain the edge widths.
            edges = list(B_prime.edges())
            edge_width = [
                B_prime.edges[edge].get('weight', EDGE_SCALE) / EDGE_SCALE
                for edge in edges
            ]
            # Step 8: Plot the local cutvertex.
            nx.draw_networkx_nodes(
                B_prime, pos,
                nodelist=[lcv.vertex],
                node_size=node_size[nodelist.index(lcv.vertex)],
                node_color=colours[0],
            )
            # Step 9: Plot the components' vertices, split vertices included.
            for i,comp in enumerate(comps):
                comp_node_list = [
                    node for node in comp
                    if not B_prime.nodes[node].get('split')
                ]
                nx.draw_networkx_nodes(
                    B_prime, pos,
                    nodelist=comp_node_list,
                    node_size=[node_size[nodelist.index(node)] for node in comp_node_list],
                    node_color=colours[i+1],
                    # edgecolors='k',
                )
                # Draw the split vertex.
                nx.draw_networkx_nodes(
                    B_prime, pos,
                    nodelist=[next(node for node,split in B_prime.nodes(data='split') if node in comp and split)],
                    node_size=min_size * 5,
                    node_shape='X',
                    node_color=colours[i+1],
                    edgecolors='k',
                )
            # Step 10: Plot the edges.
            remaining_edges = set(B_prime.edges())
            for i,comp in enumerate(comps):
                comp_edges = nx.subgraph_view(B_prime, filter_node=lambda node: node in comp).edges()
                nx.draw_networkx_edges(
                    B_prime, pos,
                    edgelist=comp_edges,
                    width=[edge_width[edges.index(edge)] for edge in comp_edges],
                    edge_color=colours[i+1],
                    alpha=0.7,
                )
                remaining_edges.difference_update(set(comp_edges))
            nx.draw_networkx_edges(
                B_prime, pos,
                edgelist=remaining_edges,
                width=[edge_width[edges.index(edge)] for edge in remaining_edges],
                alpha=0.5,
            )
            # Step 11: Plot the labels.
            nx.draw_networkx_labels(
                B_prime, pos,
                labels=labels,
            )
            # Step 12: Give it a title.
            plt.title(f'${lcv.locality}$-local cutvertex {lss(lcv.vertex.decode("utf-8"))}')
            # Step 12: Show the results.
            plt.tight_layout()
            plt.show()
            good: bool = input('Good [y/n]?\n>>> ').lower().startswith('y')

def SO_LOCAL_CUTVERTICES_BALLS_latex():
    for pic in "apache asp.net-web-api azure django ios java jenkins jquery mongodb mysql nginx osx redis".split():
        print(r"\begin{figure}[H]")
        print('\t'+r'\centering')
        print('\t'+rf'\includegraphics[width=\textwidth]{{stackoverflow/{pic}.png}}')
        print('\t'+rf'\caption{{Components of the punctured ball around the local cutvertex {pic}}}')
        print('\t'+rf'\label{{fig:stackoverflow_{pic}}}')
        print(r'\end{figure}')
        print()

def MORN_LOCAL_CUTVERTICES_RADII():
    G = __get_MORN()
    local_cutvertices = __get_pickled_local_cutvertices(G)
    fig, ax = plt.subplots()
    radii = [lcv.locality for lcv in local_cutvertices]
    bins = [
        3, 5, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        30, 35, 40, 45, 50, 250, 500, 750, 1000, 1250, 1738, 1739
    ]
    if len(bins) == 26:
        # Visually distinct colours obtained specifically for 25 bins:
        colours = [
            '#696969', '#556b2f', '#483d8b', '#b22222', '#008000',
            '#b8860b', '#000080', '#9acd32', '#8b008b', '#ff4500',
            '#ffff00', '#7cfc00', '#8a2be2', '#00ff7f', '#00bfff',
            '#0000ff', '#d8bfd8', '#ff00ff', '#1e90ff', '#db7093',
            '#f0e68c', '#ff1493', '#ffa07a', '#ee82ee', '#7fffd4'
        ]
        assert len(colours) == len(bins) - 1, 'b ru h'
    else:
        # I've modified the number of bins, just get a subset of those
        # I already have and mix and mingle.
        try:
            colours = visually_distinct_colours(len(bins)-1)
            random.shuffle(colours)
        except NotImplementedError:
            # No pretty colours then.
            colours = 'Default blue'
    hist, edges = np.histogram(radii, bins=bins)
    labels = ax.bar(
        range(len(bins)-1),
        hist,
        width=1,
        color=colours,
        edgecolor='k',
        align='edge'
    )
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.bar_label(labels)
    ax.set_xlabel('local cutvertex radius $r$')
    ax.set_ylabel('number of $r$-local cutvertices')
    fig.suptitle('Distribution of local cutvertex radii in the MORN dataset')
    plt.show()

def MORN_1739_LOCAL(pdisting: bool=False):
    from scipy.spatial.distance import pdist
    R = 1739
    G = __get_MORN()
    local_cutvertices = __get_pickled_local_cutvertices(G)
    chosen = [lcv for lcv in local_cutvertices if lcv.locality == R]
    chosen_v = [lcv.vertex for lcv in chosen]
    chosen_comp = set(nx.node_connected_component(G, chosen_v[0]))
    pos = G.graph['pos']
    if pdisting:
        pos = {lcv.vertex: pos[lcv.vertex] for lcv in chosen}
        pos = np.array(list(pos.values()))
        pdists = pdist(pos)
        fig, ax = plt.subplots()
        ax.hist(pdists, bins=20)
        plt.show()
        return
    # Not pdisting
    fig, ax = plt.subplots()
    fig.set_size_inches(*A4)
    ax.set_facecolor('xkcd:pale green')
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=chosen_v,
        node_size=1,
        node_color='xkcd:coral',
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=chosen_comp - set(chosen_v),
        node_size=1,
        node_color='xkcd:light blue',
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        ax=ax
    )
    plt.tight_layout()
    d = PROJECT_ROOT.parent / 'Thesis' / 'morn-triangle-of-death-2.png'
    plt.savefig(path_to_str(d), dpi=600)
    # plt.show()

def MORN_1739_ROAD_NAMES():
    R = 1739
    G = __get_MORN()
    local_cutvertices = __get_pickled_local_cutvertices(G)
    chosen = [lcv for lcv in local_cutvertices if lcv.locality == R]
    chosen_v = {lcv.vertex for lcv in chosen}
    chosen_comp = set(nx.node_connected_component(G, next(iter(chosen_v))))
    file = PROJECT_ROOT / 'datasets' / 'MajorOpenRoadNetwork' / 'Major_Road_Network_2018_Open_Roads.zip'
    G.graph['node_to_road'] = dict()
    with shapefile.Reader(path_to_str(file)) as shp:
        index: int = 0
        for shprec in shp.shapeRecords():
            offset: int = len(shprec.shape.points)
            section = set(range(index, index + offset))
            G.graph['node_to_road'].update(
                {v: shprec.record.name1 for v in chosen_v.intersection(section)}
            )
            index += offset
    # Plot roads and shit.
    fig, ax = plt.subplots()
    fig.set_size_inches(*A4)
    unique_roads = set(G.graph['node_to_road'].values())
    colours = [
        '#696969', '#8b4513', '#808000', '#483d8b', '#008000', '#000080',
        '#9acd32', '#8b008b', '#b03060', '#ff0000', '#ffa500', '#ffff00',
        '#7fff00', '#8a2be2', '#00ff7f', '#dc143c', '#00bfff', '#0000ff',
        '#ff7f50', '#ff00ff', '#1e90ff', '#eee8aa', '#add8e6', '#ff1493',
        '#ee82ee', '#7fffd4', '#ffc0cb'
    ]
    assert len(colours) == len(unique_roads)
    pos = G.graph['pos']
    chosen_ones = [
        {k for k,v in G.graph['node_to_road'].items() if v == road}
        for road in unique_roads
    ]
    for road, chosen_few, colour in zip(unique_roads, chosen_ones, colours):
        # nx.draw_networkx_nodes(
        #     G, pos,
        #     nodelist=chosen_few,
        #     node_size=1,
        #     node_color=colour,
        #     ax=ax,
        #     label=road if road else 'UNIDENTIFIED'
        # )
        nx.draw_networkx_edges(
            G, pos,
            edgelist=list(
                filter(
                    lambda e: e[0] in chosen_few and e[1] in chosen_few,
                    G.edges()
                )
            ),
            edge_color=colour,
            width=3,
            label=road if road else 'UNIDENTIFIED',
            ax=ax
        )
    plt.tight_layout()
    plt.legend(scatterpoints=1, loc='upper right', markerscale=20)
    d = PROJECT_ROOT.parent / 'Thesis' / 'morn-road-names.png'
    plt.show()
    # plt.savefig(path_to_str(d), dpi=600)

def INFPOWER_PLOT(G: nx.Graph=None, layout: callable=None):
    if G is None:
        G = infpower['inf-power']
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    ax.set_facecolor('xkcd:midnight blue')
    if layout is None:
        pos = __get_infpower_kamada_kawai_layout()
    else:
        pos = layout(G)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=1,
        node_color='yellow',
        node_shape='x',
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, edge_color='w', alpha=0.2, ax=ax)
    plt.show()

def INFPOWER_COMPARISON():
    G = infpower['inf-power']
    graphs = __get_Network_Data_MJS20_graphs()
    graphs.extend([stackoverflow['stackoverflow'], __get_MORN()])
    annotate_me = []
    comp_size = []
    largest_r = []
    threshold = 10_000
    for graph in graphs:
        lcvs = __get_pickled_local_cutvertices(graph)
        for comp in nx.connected_components(graph):
            chosen_few = [lcv for lcv in lcvs if lcv.vertex in comp]
            if not chosen_few:
                # No local cutvertex in component.
                continue
            n_comp = len(comp)
            r = max(lcv.locality for lcv in chosen_few)
            if n_comp >= threshold:
                annotate_me.append((lss(graph.name.stem) if 'roads' not in graph.name.stem.lower() else 'MORN', n_comp, r))
            else:
                comp_size.append(n_comp)
                largest_r.append(r)
    # Plot data.
    fig, ax = plt.subplots()
    ax.scatter(comp_size, largest_r, marker='.')
    # Plot annotated outliers.
    ax.scatter(
        list(map(operator.itemgetter(1), annotate_me)),
        list(map(operator.itemgetter(2), annotate_me)),
        marker='.', c='g'
    )
    for i, txt in enumerate(map(operator.itemgetter(0), annotate_me)):
        ax.annotate(txt, (annotate_me[i][1], annotate_me[i][2]))
    # Plot infpower
    x, y = G.number_of_nodes(), max(lcv.locality for lcv in __get_pickled_local_cutvertices(G))
    ax.scatter(x, y, marker='x', color='r')
    ax.annotate('infpower', (x, y), xytext=(x+250, y-50), xycoords='data')
    # Adjust x-axis scale.
    ticks = ax.get_xticks()
    ax.set_xscale('log')
    # ax.set_xticklabels(list(map(int, ticks)))
    ax.minorticks_off()
    # Show me the results.
    ax.set_ylabel('largest local cutvertex radius $r$')
    ax.set_xlabel('size of component of local cutvertex')
    plt.show()

def __SPLIT_COMPONENT_SIZES(G: nx.Graph, bins: List[int]=None, colour: str=None):
    if bins is None or colour is None:
        raise ValueError('m8 you are taking the piss')
    # Get size of connected components.
    comps = list(nx.connected_components(G))
    sizes = list(map(len, comps))
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.tick_params(left=False)
    ax.set_yticklabels([])
    ax.minorticks_off()
    hist, edges = np.histogram(sizes, bins=bins)
    labels = ax.bar(
        range(len(bins)-1),
        hist,
        width=1,
        color=colour,
        edgecolor='k',
        align='edge'
    )
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.bar_label(labels)
    ax.set_xlabel('size of connected component of $X^\prime$')
    ax.set_ylabel('occurrence of size in $X^\prime$')
    plt.show()

def INFPOWER_SPLIT_COMPONENT_SIZES():
    G = __get_infpower_split()
    to_remove = [node for (node, split) in G.nodes(data='split') if split]
    G.remove_nodes_from(to_remove)
    bins = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        25, 50, 75, 125, 200, 201
    ]
    colour = 'xkcd:bright yellow'
    __SPLIT_COMPONENT_SIZES(G, bins, colour)

def MORN_SPLIT_COMPONENT_SIZES():
    # Get MORN split.
    G = __get_MORN()
    lcvs = __get_pickled_local_cutvertices(G)
    split_at_local_cutvertices(G, lcvs, inplace=True)
    to_remove = [node for node, split in G.nodes(data='split') if split]
    G.remove_nodes_from(to_remove)
    # YUH
    bins = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        25, 50, 75, 125, 200, 2000
    ]
    colour = 'xkcd:light olive green'
    __SPLIT_COMPONENT_SIZES(G, bins, colour)

def __SPLIT_COMPONENT_ISOMORPHISM_TEST(G: nx.Graph, k: int):
    assert k == 3 or  k == 4 or k == 5
    if k == 3:
        func = __get_connected_graphs_3_vertices
    elif k == 4:
        func = __get_connected_graphs_4_vertices
    elif k == 5:
        func = __get_connected_graphs_5_vertices
    else:
        raise ValueError('maaaAAte')
    graphs: List[nx.Graph] = func()
    # Obtain components of size k.
    components = list(
        filter(
            lambda comp: len(comp) == k,
            nx.connected_components(G)
        )
    )
    print(len(components), 'to check...')
    # Check for isomorphism.
    iso = []
    for comp in components:
        for i, graph in enumerate(graphs):
            if nx.is_isomorphic(nx.subgraph_view(G, filter_node=lambda node: node in comp), graph):
                iso.append(i)
                break
    # Show me.
    c = Counter(iso)
    print(c)
    if not input('Plot this shit? [y/n]').lower().startswith('y'):
        print('Good call.')
        return
    # Plot the isomorphic graphs.
    dims = (1, len(c)) if len(c) < 10 else (3, math.ceil(len(c)/3))
    fig, axes = plt.subplots(*dims)
    try:
        axes = axes.reshape(-1)
    except AttributeError:
        axes = [axes]
    layout = nx.kamada_kawai_layout
    for (i, count), ax in zip(c.items(), axes):
        ax.set_title(f'Graph {i} appears {pluralise(count, "time")}')
        G = graphs[i]
        nx.draw_networkx(G, layout(G), ax=ax)
    plt.show()

def INFPOWER_SPLIT_COMPONENT_ISOMORPHISM_TEST(k: int):
    # Get infpower split.
    G = __get_infpower_split()
    to_remove = [node for (node, split) in G.nodes(data='split') if split]
    G.remove_nodes_from(to_remove)
    # Do the thing.
    __SPLIT_COMPONENT_ISOMORPHISM_TEST(G, k)

def MORN_SPLIT_COMPONENT_ISOMORPHISM_TEST(k: int):
    # Get MORN split.
    G = __get_MORN()
    lcvs = __get_pickled_local_cutvertices(G)
    print('Splitting...')
    split_at_local_cutvertices(G, lcvs, inplace=True)
    print('Done!')
    to_remove = [node for node, split in G.nodes(data='split') if split]
    G.remove_nodes_from(to_remove)
    # Do the thing.
    __SPLIT_COMPONENT_ISOMORPHISM_TEST(G, k)

# QUICK PROTOTYPING (lol, find an accurate name)

def split_vertices():
    # Construct G
    rim = nx.cycle_graph(range(1, 13))
    hub = '$v$'
    spokes = []
    for i in range(1, 13, 4):
        spokes.extend([(hub, i), (hub, i+1)])
    G = rim.copy()
    G.add_edges_from(spokes)
    # Split at the hub with radius 3.
    radius = 3
    H = split_at_vertices(G, hub, radius, inplace=False)
    # Obtain the positions.
    pos_G = nx.kamada_kawai_layout(G)
    pos_H = nx.kamada_kawai_layout(H)
    # Associate a colour to each local component tag.
    colours = 'y', 'k', 'g', 'b'
    nodelists: Dict[int, List[Vertex]] = {}
    v: Vertex
    tags: Set[int]
    for v, tags in H.nodes(data='local_component_tag'):
        if isinstance(v, str):
            continue
        if tags is not None:
            for tag in tags:
                try:
                    nodelists[tag].append(v)
                except KeyError:
                    nodelists[tag] = [v]
        else:
            try:
                nodelists[tags].append(v)
            except KeyError:
                nodelists[tags] = [v]
    # assert len(nodelists) == len(colours), f'have {len(nodelists)} tags but {len(colours)} colours'
    # Plot shit.
    fig, axes = plt.subplots(1, 3)
    ## axes[0] is for G, axes[1] is for G with the ball, axes[2] is for H

    ## Plot G first
    ax = axes[0]
    ### plot hub
    nx.draw_networkx_nodes(G, pos_G, nodelist=[hub], node_color='r', label=hub, ax=ax)
    ### plot rest
    nx.draw_networkx_nodes(G, pos_G, nodelist=G.nodes() - {hub}, node_color='k', label=f'NOT {hub}', ax=ax)
    nx.draw_networkx_edges(G, pos_G, alpha=0.1, ax=ax)
    nx.draw_networkx_labels(G, pos_G, font_color='w', ax=ax)
    ax.legend(scatterpoints=1, loc='best')

    ## Plot G with ball highlighted
    ax = axes[1]
    ### obtain ball
    ball_G = ball(G, hub, radius/2)
    ball_nodes = ball_G.nodes()
    ### draw nodes
    nx.draw_networkx_nodes(G, pos_G, nodelist=[hub], node_color='r', label=hub, ax=ax)
    nx.draw_networkx_nodes(G, pos_G, nodelist=ball_nodes - {hub}, node_color='g', label=f'$B_{{3/2}}({hub.replace("$", "")})-{hub.replace("$", "")}$', ax=ax)
    nx.draw_networkx_nodes(G, pos_G, nodelist=G.nodes() - ball_nodes, node_color='k', label='other nodes', ax=ax)
    nx.draw_networkx_edges(G, pos_G, edgelist=ball_G.edges(), ax=ax)
    nx.draw_networkx_edges(G, pos_G, edgelist=G.edges() - ball_G.edges(), alpha=0.1, ax=ax)
    nx.draw_networkx_labels(G, pos_G, font_color='w', ax=ax)
    ax.legend(scatterpoints=1, loc='best')
    
    ## Plot H last
    ax = axes[2]
    ### plot local component tagged
    local_component_edges: Tuple = []
    for (tag, nodelist), colour in zip(nodelists.items(), colours):
        nx.draw_networkx_nodes(H, pos_H, nodelist=nodelist, node_color=colour, label='tagged' if tag else 'no tag', ax=ax)
        if tag is not None:
            edges = [(x,y) for (x,y) in H.edges() if x in nodelist and y in nodelist]
            nx.draw_networkx_edges(H, pos_H, edgelist=edges, edge_color=colour, alpha=0.2, width=4, ax=ax)
            local_component_edges.extend(edges)
    ### plot split vertices
    nx.draw_networkx_nodes(H, pos_H, nodelist=[v for v,split in H.nodes(data='split') if split], node_color='purple', label='split vertex', ax=ax)
    nx.draw_networkx_nodes(H, pos_H, nodelist=[hub], node_color='r', label=hub, ax=ax)
    nx.draw_networkx_edges(H, pos_H, edgelist=H.edges() - set(local_component_edges), alpha=0.1, ax=ax)
    nx.draw_networkx_labels(H, pos_H, font_color='w', ax=ax)
    ax.legend(scatterpoints=1, loc='best')
    plt.show()

def is_this_definitely_working():
    # Construct G.
    G: nx.Graph = nx.Graph()
    v_edges: List[Tuple[Vertex, Vertex]] = [(x, 'v') for x in 'abc']
    w_edges: List[Tuple[Vertex, Vertex]] = [(x, 'w') for x in 'def']
    G.add_edges_from(v_edges)
    G.add_edges_from(w_edges)
    G.add_edge('v', 'w')
    # Construct an artificial list of local cutvertices.
    v_locality: int = 2
    w_locality: int = 2
    local_cutvertices: List[LocalCutvertex] = [
        LocalCutvertex(vertex='v', locality=v_locality, edge_partition={('w',), tuple('abc')}),
        LocalCutvertex(vertex='w', locality=w_locality, edge_partition={('v',), tuple('def')}),
    ]
    # Split at the local cutvertices to obtain H.
    H: nx.Graph = split_at_local_cutvertices(G, local_cutvertices, inplace=False)
    # Delete the local cutvertices from H to obtain H_prime.
    H_prime: nx.Graph = H.copy()
    H_prime.remove_nodes_from('vw')
    # Visualise the results.
    fig, axes = plt.subplots(1, 3)
    graphs: Tuple[nx.Graph] = (G, H, H_prime)
    names: Tuple[str] = ('G', 'H', r'H^\prime')
    for graph, name, ax in zip(graphs, names, axes.reshape(-1)):
        nx.draw_networkx(graph, with_labels=True, ax=ax)
        ax.set_title(f'Graph ${name}$')
    plt.show()

def flc(G: nx.Graph, checkpoint_file: Path, min_locality: int=3, every: int=100) -> List[LocalCutvertex]:
    '''
        src.find_local_cutvertices with checkpointing.
    '''
    components: List[Set[Vertex]] = list(nx.connected_components(G))
    if checkpoint_file.exists() and checkpoint_file.stat().st_size > 0:
        # Checkpoint file exists.
        with open(checkpoint_file, 'rb') as handle:
            data = pickle.load(handle)
            try:
                checkpoint: int
                local_cutvertices: List[LocalCutvertex]
                checkpoint, local_cutvertices = data
                print(f'<FLC> [{checkpoint}/{len(G.nodes())}] Checkpoint found!')
            except ValueError:
                print('<FLC> Routine previously ran to completion, returning result...')
                return data
    else:
        # Checkpoint file does not exist.
        checkpoint: int = 0
        local_cutvertices: List[LocalCutvertex] = []
    # Apply checkpoint.
    nodes: List[Vertex] = list(G.nodes())
    nodes = nodes[checkpoint:]
    start: float = time.perf_counter()
    for i, v in enumerate(nodes):
        # Save me progress matey.
        if i and not i % every:
            now: float = time.perf_counter()
            print(f'<FLC> [{checkpoint+i}/{len(nodes)+checkpoint}] Processed {pluralise(every, "vertex")} in {sec2str(now-start)}, found {pluralise(len(local_cutvertices), "local cutvertex")} so far...' + ' '*20 + '\r', end='')
            with open(checkpoint_file, 'wb') as handle:
                pickle.dump((checkpoint+i, local_cutvertices), handle)
            start = now
        # dors wesh gros t'es mort
        mi: int = min_locality
        component_vertices: Set[Vertex] = next(comp for comp in components if v in comp)
        ma: int = len(component_vertices)
        if mi > ma:
            continue
        component_without_v: nx.Graph = nx.classes.graphviews.subgraph_view(
            G, filter_node=lambda node: node in component_vertices and node != v
        )
        if not nx.algorithms.components.is_connected(component_without_v):
            continue
        component: nx.Graph = nx.classes.graphviews.subgraph_view(
            G, filter_node=lambda node: node in component_vertices
        )
        mid: int = None
        v_is_a_local_cutvertex: bool = None
        while True:
            if ma - mi == 1 and v_is_a_local_cutvertex is not None:
                if not v_is_a_local_cutvertex:
                    v_is_a_local_cutvertex: bool = is_local_cutvertex(component, v, mi)
                    mid: int = mi
                    break
                else:
                    v_is_a_local_cutvertex: bool = is_local_cutvertex(component, v, ma)
                    if v_is_a_local_cutvertex:
                        mid: int = ma
                    else:
                        v_is_a_local_cutvertex: bool = True
                    break
            else:
                mid: int = mi + math.floor((ma - mi) / 2)
            v_is_a_local_cutvertex: bool = is_local_cutvertex(component, v, mid)
            if mi == ma:
                break
            if v_is_a_local_cutvertex:
                mi: int = mid
            else:
                ma: int = mid
        if v_is_a_local_cutvertex:
            punctured_ball: nx.Graph = nx.classes.graphviews.subgraph_view(
                ball(G, v, mid/2), filter_node=lambda x: x != v
            )
            punctured_ball_components: Generator[Set[Vertex], None, None] = nx.connected_components(punctured_ball)
            neighbourhood: Set[Vertex] = set(G.neighbors(v))
            edge_partition: Set[Tuple[Vertex, ...]] = set(
                tuple(neighbourhood.intersection(comp)) for comp in punctured_ball_components
            )
            local_cutvertices.append(
                LocalCutvertex(vertex=v, locality=mid, edge_partition=edge_partition)
            )
    print('\n<FLC> Done finding local cutvertices, saving result...')
    with open(checkpoint_file, 'wb') as handle:
        pickle.dump(local_cutvertices, handle)
    print(f'<FLC> Result saved ({humanize.naturalsize(checkpoint_file.stat().st_size)}).')
    return local_cutvertices

# STATS SHIT

def local_cutvertex_radii_distribution(G: nx.Graph, min_locality: int=None, thresholds: bool=True, weird: bool=False):
    with open(_pickle_name(G.name), 'rb') as handle:
        local_cutvertices: List[LocalCutvertex] = pickle.load(handle)
    c = Counter(lcv.locality for lcv in local_cutvertices)
    if min_locality is not None:
        c = {k: v for k,v in c.items() if v >= min_locality}
    keys, vals = c.keys(), c.values()
    vals = list(vals)
    total: int = sum(vals)
    fig, ax = plt.subplots(1, 1)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    if weird:
        ax.set_xscale('log')
        ax.set_xticks(list(keys))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.bar(keys, vals)
    if thresholds:
        if isinstance(thresholds, list):
            ax.set_yticks(list(ax.get_yticks()) + thresholds)
        else:
            ax.set_yticks(list(vals))
    threshold_line_params = {
        'linewidth': 1,
        'color': 'black',
        'linestyle': 'dashed',
        'alpha': 0.25
    }
    if thresholds:
        if isinstance(thresholds, list):
            vals = list(ax.get_yticks())
        for val in vals:
            ax.axhline(y=val, **threshold_line_params)
            
    ax.set_xlabel('local cutvertex radius $r$')
    ax.set_ylabel('number of $r$-local cutvertices')
    ax.set_title(
        'Local cutvertex radii distribution for '
        + lss(G.name.name)
        + f' ({pluralise(total, "local cutvertex")})'
    )
    plt.tight_layout()
    plt.show()

def number_of_components_post_splitting_NDMJS20():
    '''
        For each graph, split at its local cutvertices, delete the local
        cutvertices in the resulting graph, and plot the number of components.
        I'm going to cycle through the different minimum localities I'd
        like to consider.
    '''
    # Obtain subplots.
    localities: List[int] = list(range(3, 10))
    fig1, axes1 = plt.subplots(1, 2)
    fig2, axes2 = plt.subplots(1, 2)
    fig3, axes3 = plt.subplots(1, 2)
    fig4, axes4 = plt.subplots(1, 2)
    axes = np.hstack([axes1, axes2, axes3, axes4])
    # Obtain graphs.
    graphs: List[nx.Graph] = __get_Network_Data_MJS20_graphs()
    # Obtain graph names.
    MAX_LENGTH: int = 10
    graph_names: List[str] = [
        escape_underscore(graph.name.stem[:MAX_LENGTH])
        for graph in graphs
    ]
    # Obtain their local cutvertices.
    local_cutvertices: List[List[LocalCutvertex]] = []
    for graph in graphs:
        with open(_pickle_name(graph.name), 'rb') as handle:
            local_cutvertices.append(pickle.load(handle))
    # For each locality, split all vertices larger than or equal to
    # that locality, and visualise the number of components.
    for locality, ax in zip(localities, axes.reshape(-1)):
        # Configure matplotlib axes.
        ax.set_xticklabels(graph_names, rotation='vertical')
        # Obtain split graphs without local cutvertices.
        split_graphs: List[nx.Graph] = []
        for graph, lcvs in zip(graphs, local_cutvertices):
            # Filter local cutvertices by locality.
            lcvs_filtered: List[LocalCutvertex] = [
                lcv for lcv in lcvs
                if lcv.locality >= locality
            ]
            lcvs_filtered_vertices: Set[Vertex] = set(
                lcv.vertex for lcv in lcvs_filtered
            )
            # Split at graph.
            split_graph: nx.Graph = split_at_local_cutvertices(
                graph,
                lcvs_filtered,
                inplace=False
            )
            # Remove local cutvertices from split graph.
            split_graph.remove_nodes_from(lcvs_filtered_vertices)
            # Add to list of split graphs.
            split_graphs.append(split_graph)
        # Find the number of connected components for each modified split graph.
        num_components: List[int] = list(
            map(nx.number_connected_components, split_graphs)
        )
        # Plot this data.
        ## Obtain interesting graphs.
        interesting_graphs: List[Tuple[str, int]] = [
            (graph_name, num_comp)
            for graph_name, num_comp in zip(graph_names, num_components)
            # if num_comp > 1
        ]
        ## Plot interesting graphs.
        interesting_names, interesting_num_components = zip(*interesting_graphs)
        # ax.set_yticks(interesting_num_components)
        ax.bar(interesting_names, interesting_num_components)
        ## Add dotted lines for readibility.
        threshold_line_params = {
            'linewidth': 0.2,
            'color': 'grey',
            'linestyle': 'dashed',
            'alpha': 0.25
        }
        for num_comp in interesting_num_components:
            ax.axhline(y=num_comp, **threshold_line_params)
        # Give a title.
        ax.set_title(f'$r \geq{locality}$')
    # Set up figure title.
    for fig in fig1, fig2, fig3, fig4:
        fig.suptitle('\# components when split at local cutvertices and removed')
    # Plot the number of components in the original graph.
    fig, ax = plt.subplots(1, 1)
    ax.set_xticklabels(graph_names, rotation=45, ha='right')
    num_components: List[int] = list(
        map(nx.number_connected_components, graphs)
    )
    ax.set_yticks(num_components)
    ax.bar(graph_names, num_components)
    ## Add dotted lines for readibility.
    threshold_line_params = {
        'linewidth': 0.2,
        'color': 'grey',
        'linestyle': 'dashed',
        'alpha': 0.25
    }
    for num_comp in interesting_num_components:
        ax.axhline(y=num_comp, **threshold_line_params)
    ## Give a title.
    ax.set_title(f'\# components original graph')
    # Show us the money.
    plt.tight_layout()
    plt.show()

def redundant_points_MORN(G: nx.Graph=None) -> List[Vertex]:
    '''
        Returns a list of redundant points in the Major_Road_Network_2018_Open_Roads dataset.

        Parameters
        ----------
        G: nx.Graph, optional
            The graph corresponding to the MORN dataset. If None, will be retrieved.
        
        Notes
        -----
        A point is redundant if it is collinear with its 2 neighbours, and non-trivial otherwise.
        
        Returns
        -------
        List[Vertex]
            The redundant points.
    '''
    if G is None:
        G: nx.Graph = __get_MORN()
    redundant: List[Vertex] = []
    for vertex, X in G.graph['pos'].items():
        if G.degree(vertex) != 2:
            continue
        Y, Z = [G.graph['pos'][x] for x in G.neighbors(vertex)]
        if collinear(X, Y, Z):
            redundant.append(vertex)
    return redundant

def remove_redundant_points(G: nx.Graph, redundant_points: List[Vertex]):
    '''
        OPERATES IN-PLACE.
    '''
    # Create edges around redundant points.
    # NOTE: a redundant point v has 2 neighbours, hence tuple(H.neighbors(v)) links its neighbours.
    G.add_edges_from([
        tuple(G.neighbors(vertex)) for vertex in redundant_points
    ])
    # Delete redundant points.
    G.remove_nodes_from(redundant_points)

def analyse_MORN(how_many: int):
    '''
        This functions runs some stats on the Major_Road_Network_2018_Open_Roads dataset,
        including:
        
        - redundant points
        - differences between redundant points and their neighbours

        Notes
        -----
        A point is redundant if it is collinear with its 2 neighbours, and non-trivial otherwise.
    '''
    # Obtain the dataset and the corresponding graph.
    dataset: str = 'Major_Road_Network_2018_Open_Roads'
    zip_file: Path = (MajorOpenRoadNetworks.dataset_folder() / dataset).with_suffix('.zip')
    G: nx.Graph = MajorOpenRoadNetworks[dataset]
    # Find the redundant points.
    redundant: List[Vertex] = redundant_points_MORN(G)
    print('There are', len(redundant), 'redundant points in the MORN dataset.')
    # Compare the information of some of the redundant points with their neighbours.
    lucky_few: List[Vertex] = random.sample(redundant, how_many)
    neighbours: List[Tuple[Vertex, Vertex]] = [
        tuple(G.neighbors(lucky)) for lucky in lucky_few
    ]
    with shapefile.Reader(path_to_str(zip_file)) as shp:
        shape_records: List[Tuple[shapefile.ShapeRecord, shapefile.ShapeRecord, shapefile.ShapeRecord]] = [
            (shp.shapeRecord(lucky), shp.shapeRecord(x), shp.shapeRecord(y))
            for lucky,(x,y) in zip(lucky_few, neighbours)
        ]
    shape_fields_to_compare = [
        'parts',
    ]
    # Choose which fields to compare.
    # Frontrunner idea is comparing the ShapeRecord.shape.parts fields, if they're distinct then the point is
    # no longer redundant imo.
    for i in range(how_many):
        data = [
            [
                getattr(shprec.shape, field) for field in shape_fields_to_compare
            ] for shprec in shape_records[i]
        ]
        for i, name in enumerate(('Redundant', 'Neighbour A', 'Neighbour B')):
            data[i].insert(0, name)
        table = tabulate.tabulate(
            data,
            headers=['Point'] + shape_fields_to_compare
        )
        print('Redundant Entry', i)
        print('='*40)
        print(table)

def compare_MORN_redundant_points():
    '''
        Plots the graph corresponding to the Major_Road_Network_2018_Open_Roads dataset
        and the MORN dataset with its redundant points removed side-by-side.
    '''
    # Obtain MORN dataset graph.
    G: nx.Graph = __get_MORN()
    # Obtain redundant points.
    redundant: List[Vertex] = redundant_points_MORN(G)
    savings_ratio: float = 1 - (G.number_of_nodes() - len(redundant)) / G.number_of_nodes()
    savings_percentage: float = round(savings_ratio * 100, 2)
    # Remove redundant points from G.
    H: nx.Graph = G.copy()
    remove_redundant_points(H, redundant)
    # Obtain matplotlib plot.
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(A4[0]*2, A4[1])
    fig.set_dpi(1200)
    # Plot G.
    nx.draw(G, G.graph['pos'], node_size=1, node_color='red', edge_color='gray', ax=axes[0])
    axes[0].set_title('Graph $G$ representing the MORN dataset')
    # Plot H.
    nx.draw(H, H.graph['pos'], node_size=1, edge_color='gray', ax=axes[1])
    axes[1].set_title(f'Graph $H$, a ${savings_percentage}$\% more lightweight representation of $G$')
    # Show me.
    plt.savefig('huh.png')

def large_plot_MORN(G: nx.Graph=None, special: LocalCutvertex=None, node_color: str='red', fname: str='large_plot.png', figure_size: FigureSize=A4, dpi: int=600, face_colour: str='xkcd:pale green'):
    if G is None:
        G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    fig, ax = plt.subplots()
    if face_colour is not None:
        fig.set_facecolor(face_colour)
    fig.set_size_inches(*figure_size)
    fig.set_dpi(dpi)
    if special is not None:
        ax.set_title(escape_underscore(G.name.stem) + f' with {special.vertex} and $B_\\frac{{{special.locality}}}{{2}}({special.vertex})$ highlighted')
        ball_around_special: nx.Graph = nx.subgraph_view(
            ball(G, special.vertex, special.locality / 2), filter_node=lambda node: node != special.vertex
        )
        nx.draw_networkx_nodes(G, pos, node_size=5, node_color='blue', nodelist=[special.vertex], ax=ax)
        nx.draw(ball_around_special, pos, node_size=0, edge_color='xkcd:red', ax=ax)
        rest: nx.Graph = nx.subgraph_view(
            G, filter_edge=lambda x,y: (x,y) not in ball_around_special.edges()
        )
        nx.draw(rest, pos, node_size=0, node_color=node_color, edge_color='gray', ax=ax)
    else:
        nx.draw(G, pos, node_size=0, node_color=node_color, ax=ax)
    if fname is None:
        plt.show()
    else:
        plt.tight_layout()
        ax.set_aspect('equal')
        plt.savefig(fname, dpi=dpi, transparent=face_colour is None)

def random_plot_MORN_components(how_many: int, G: nx.Graph=None):
    if G is None:
        G: nx.Graph = __get_MORN()
    # Obtain layout.
    pos = G.graph['pos']
    # Obtain components.
    components: List[Set[Vertex]] = list(nx.connected_components(G))
    # Sanity check.
    assert how_many <= len(components), 'b r u h'
    # See that I have enough colours.
    try:
        node_colours = visually_distinct_colours(how_many)
    except NotImplementedError:
        raise ValueError(f'cannot plot {how_many} components, sorry')
    # Sort out number of rows and columns.
    square_root: float = math.sqrt(how_many)
    decimal, integral = math.modf(square_root)
    if not decimal:
        nrows = ncols = int(square_root)
    # Want a long rectangle plot, not a tall one.
    elif decimal < 0.5:
        nrows: int = int(integral)
        ncols: int = math.ceil(square_root)
    else:
        nrows: int = int(integral)
        ncols: int = math.ceil(square_root) + 1
    # Obtain figure and axes.
    fig, axes = plt.subplots(nrows, ncols)
    fig.suptitle(f'${how_many}$ randomly chosen components from the MORN dataset')
    axes = axes.reshape(-1)
    # Randomly choose components.
    components_to_highlight: List[Set[Vertex]] = random.sample(components, how_many)
    # Plot.
    for ax, component_to_highlight, node_colour in zip(axes, components_to_highlight, node_colours):
        ax.set_title(f'${len(component_to_highlight)}$ vertices')
        component_edges_to_highlight: List[Tuple[Vertex, Vertex]] = [
            (x,y) for x,y in G.edges()
            if x in component_to_highlight and y in component_to_highlight
        ]
        nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=component_to_highlight, node_color=node_colour, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=component_edges_to_highlight, edge_color='k', ax=ax)
    # Show me the money.
    plt.show()

def leaves_inter_node_distance_MORN(G: nx.Graph=None):
    from itertools import chain
    from scipy.spatial.distance import pdist

    if G is None:
        G: nx.Graph = __get_MORN()
    pos = G.graph['pos']

    leaves: List[Vertex] = [node for node in G.nodes() if G.degree(node) == 1]
    positions: List[Tuple[float, float]] = [pos[leaf] for leaf in leaves]

    # Imagine a square matrix of the inter-leaf distances in G.
    # I want the upper triangular part of that matrix, definitely
    # excluding the diagonal.
    print('Computing distances...')
    points: np.ndarray = np.array(positions)
    start: float = time.perf_counter()
    distances: np.ndarray = pdist(points)
    end: float = time.perf_counter()
    print(distances.shape)
    exit()
    distances = distances.reshape(-1)
    print(f'Computed {pluralise(len(distances), "pairwise distance")} in {sec2str(end-start)}.')
    # Plot distances.
    bins = [0, .25, .5, .75, 1, 10, 100, 1_000, 10_000, 100_000, 750_000]
    plt.xscale('log')
    plt.yscale('log')
    print('Plotting distances...')
    start: float = time.perf_counter()
    plt.hist(distances, bins=bins)
    end: float = time.perf_counter()
    print(f'Plotted data in {sec2str(end-start)}.')
    plt.show()

def closest_non_component_leaf_MORN(G: nx.Graph=None, print_every: int=500, bins: int=50) -> Dict[Vertex, Tuple[Vertex, float]]:
    # If a pickle result exists, use that, if not go ahead with the whole shabang.
    filename: Path = PROJECT_ROOT / 'closest_non_component_leaf.pickle'

    if not (filename.exists() and filename.stat().st_size > 0):
        from scipy.spatial.distance import pdist
        from scipy.special import comb

        if G is None:
            G: nx.Graph = __get_MORN()
        pos = G.graph['pos']

        leaves: np.ndarray = np.array([node for node in G.nodes() if G.degree(node) == 1])
        print(f'Processing {pluralise(len(leaves), "leaf")}...')
        points: np.ndarray = np.array([pos[leaf] for leaf in leaves])

        # Compute distances.
        print('Computing distances...')
        start: float = time.perf_counter()
        distance_vector: np.ndarray = pdist(points)
        end: float = time.perf_counter()
        print(f'Computed {pluralise(len(distance_vector), "pairwise distance")} in {sec2str(end-start)}.')
        
        # For each leaf, find its nearest leaf neighbour that isn't in the same component.
        m: int = len(points)
        def get_distance(i: int, j: int) -> float:
            '''Returns the distance between the leaves at indices i and j'''
            # If identical indices then 0.
            if i == j:
                return 0
            # Require i < j.
            i, j = min(i, j), max(i, j)
            # Refer to: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            idx: int = m * i + j - ((i + 2) * (i + 1)) // 2
            return distance_vector[idx]
        print('Obtaining connected components...')
        components: List[Set[Vertex]] = list(nx.connected_components(G))
        component_id: Dict[Vertex, int] = {
            v: i
            for i in range(len(components))
            for v in components[i]
        }
        components.clear()
        print('Obtained connected components.')
        closest_non_component_leaf: Dict[Vertex, Tuple[Vertex, float]] = {}
        print('Finding closest non-component leaves...')
        overall_start: float = time.perf_counter()
        start: float = time.perf_counter()
        for i, leaf in enumerate(leaves):
            if i and not (i % print_every):
                now: float = time.perf_counter()
                print(f'Processed {pluralise(i, "leaf")} in {sec2str(now - start)}.')
                start = now
            # Obtain the distances from the current leaf to all other leaves.
            distances: np.ndarray = np.array([get_distance(i, j) for j in range(m)])
            # Argsort the distances in ascending order.
            arg_sort: np.ndarray = np.argsort(distances)
            # Use arg_sort on the leaves.
            for other_leaf, distance in zip(leaves[arg_sort], distances[arg_sort]):
                # Find the closest non-component leaf.
                if leaf == other_leaf:
                    continue
                if component_id[leaf] == component_id[other_leaf]:
                    continue
                # FOUND!
                closest_non_component_leaf[leaf] = (other_leaf, distance)
        overall_end: float = time.perf_counter()
        print(f'Found closest non-component leaves in {sec2str(overall_end - overall_start)}.')

        # Save result.
        filename: Path = PROJECT_ROOT / 'closest_non_component_leaf.pickle'
        print('Saving result...')
        with open(filename, 'wb') as handle:
            pickle.dump(closest_non_component_leaf, handle)
        print(f'Saved result ({humanize.naturalsize(filename.stat().st_size)}).')
    else:
        print(f'Loading saved result ({humanize.naturalsize(filename.stat().st_size)})...')
        with open(filename, 'rb') as handle:
            closest_non_component_leaf: Dict[Vertex, Tuple[Vertex, float]] = pickle.load(handle)
        print('Loaded saved result.')

    # Plot the distribution of closest non-component leaf distances.
    # print('Plotting distribution of closest non-component leaf distances...')
    # start: float = time.perf_counter()
    # plt.hist([d for _,d in closest_non_component_leaf.values()], bins=bins)
    # end: float = time.perf_counter()
    # print(f'Plotted distribution in {sec2str(end-start)}.')
    # plt.show()
    return closest_non_component_leaf

def closest_non_component_vertex_MORN(G: nx.Graph=None, plotting: bool=False, save_every: int=200):
    if G is None:
        G: nx.Graph = __get_MORN()
    pos = G.graph['pos']

    nodes: List[Vertex] = list(G.nodes())
    filename: Path = PROJECT_ROOT / 'MORN' / f'closest_non_component_vertex.pickle'
    if not (filename.exists() and filename.stat().st_size > 0):
        # I need to find the closest non-component vertex to each vertex.
        # Approach A: let's first compute the distances from vertex to all non-component vertices.
        print('<CNCV> No savepoint found, starting from scratch...')
        checkpoint_vertex: int = 0
        closest_non_component_vertices: Dict[Vertex, Tuple[Vertex, float]] = {}
    else:
        print('<CNCV> Savepoint found! Loading...')
        with open(filename, 'rb') as handle:
            checkpoint_vertex: int
            closest_non_component_vertices: Dict[Vertex, Tuple[Vertex, float]]
            try:
                data = pickle.load(handle)
                checkpoint_vertex, closest_non_component_vertices = data
            except ValueError:
                # Assume the routine has run until completion in the past.
                print('<CNCV> Routine previously successfully completed.')
                closest_non_component_vertices = data
                return closest_non_component_vertices
        print(f'<CNCV> [{checkpoint_vertex}/{len(nodes)}] Savepoint loaded.')
    start: float = time.perf_counter()
    for i, vertex in enumerate(nodes[checkpoint_vertex:]):
        if i and not i % save_every:
            now: float = time.perf_counter()
            with open(filename, 'wb') as handle:
                pickle.dump((checkpoint_vertex + i, closest_non_component_vertices), handle)
            print(' '*100 + '\r', end='')
            print(f'[{checkpoint_vertex + i}/{len(nodes)}] Processed {save_every} vertices in {sec2str(now - start)} ...\r', end='')
            start = now
        # Find the vertex's component.
        component: Set[Vertex] = nx.node_connected_component(G, vertex)
        # Store all the other vertices.
        vertices: np.ndarray = np.array([node for node in G.nodes() - component])
        points: np.ndarray = np.array([pos[node] for node in vertices])
        # Duplicate vertex's Cartesian coordinates.
        vertex_pos: np.ndarray = np.array([pos[vertex]])
        vertex_pos: np.ndarray = np.vstack([vertex_pos for _ in range(points.shape[0])])
        # Compute the distance from vertex to all the other vertices.
        distances: np.ndarray = np.linalg.norm(points - vertex_pos, axis=1)
        # np.argsort to find the closest non-component vertex.
        argsort: np.ndarray = np.argsort(distances)
        distance: float = distances[argsort][0]
        closest_non_component_vertex: Vertex = vertices[argsort][0]
        # Add to dictionary.
        closest_non_component_vertices[vertex] = (closest_non_component_vertex, distance)
    print('Saving result...')
    with open(filename, 'wb') as handle:
        pickle.dump(closest_non_component_vertices, handle)
    print(f'Saved result ({humanize.naturalsize(filename.stat().st_size)}).')
    
    # Plot the closest non-component vertex and vertex of interest on the big graph.
    if plotting:
        raise NotImplementedError('b r u h')
        fig, ax = plt.subplots()
        ax.set_title(f'Distance: ${distance}$')
        nx.draw_networkx_nodes(G, pos, nodelist=G.nodes() - {vertex, closest_non_component_vertex}, node_size=1, alpha=0.1, ax=ax)
        for node, colour, size in zip([vertex, closest_non_component_vertex], ['red', 'green'], [20, 10]):
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size, node_color=colour, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.1, width=0.1, ax=ax)
        plt.show()

def overlapping_groups_MORN(G: nx.Graph=None, save_every: int=200) -> List[Set[Vertex]]:
    if G is None:
        G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    filename: Path = PROJECT_ROOT / 'MORN' / 'overlapping_groups.pickle'
    closest_non_component_vertices: Dict[Vertex, Tuple[Vertex, float]] = closest_non_component_vertex_MORN(G)
    overlapping_points: List[Tuple[Vertex, Vertex]] = [
        (k,v) for k,(v,d) in closest_non_component_vertices.items() if not d
    ]
    if not (filename.exists() and filename.stat().st_size > 0):
        # No checkpoint, start from scratch.
        checkpoint: int = 0
        groups: List[Set[Vertex]] = []
    else:
        # Potential checkpoint, try loading.
        checkpoint: int
        groups: List[Set[Vertex]]
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            try:
                checkpoint, groups = data
                print(f'<OG> [{checkpoint}/{len(overlapping_points)}] Checkpoint found!')
            except ValueError:
                print('<OG> Routine previously ran to completion, returning result...')
                return data
    for i, (group_coordinator,closest) in enumerate(overlapping_points[checkpoint:]):
        # Begin by checking if a group has already been created for either the group
        # coordinator or its closest vertex.
        try:
            group: Set[Vertex] = next(
                group for group in groups
                if group_coordinator in group or closest in group
            )
            # Group found, add both to the group (set.add deals with potential redundancy).
            group.add(group_coordinator)
            group.add(closest)
            # Carry on.
            continue
        except StopIteration:
            # A group entry hasn't been made for either vertices under consideration,
            # proceed with the routine and create one for them.
            pass
        # Are we saving the result (and simultaneously, outputting information)?
        debug: bool = len(groups) and not len(groups) % save_every
        if debug:
            print(f'Creating group #{len(groups)+1}...' + ' '*30 + '\r', end='')
        # Create starting group.
        group: Set[Vertex] = {group_coordinator, closest}
        # Go through vertices hoping to expand the group.
        for vertex in closest_non_component_vertices:
            if pos[vertex] == pos[group_coordinator]:
                group.add(vertex)
        # Add group to list of groups.
        groups.append(group)
        # Save progress if necessary.
        if debug:
            with open(filename, 'wb') as handle:
                pickle.dump((i, groups), handle)
    print('<OG> Overlapping groups of points established, saving global result...')
    with open(filename, 'wb') as handle:
        pickle.dump(groups, handle)
    print(f'<OG> Saved result ({humanize.naturalsize(filename.stat().st_size)}).')
    print('<OG> Returning result...')
    return groups

def flatten_MORN_using_cncv(G: nx.Graph=None, every: int=100) -> Tuple[List[int], nx.Graph]:
    # Obtain MORN.
    if G is None:
        G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    # Obtain overlapping groups of vertices.
    groups: List[Set[Vertex]] = overlapping_groups_MORN(G=G)
    # Check for checkpoint.
    filename: Path = PROJECT_ROOT / 'MORN' / 'flattened_MORN.pickle'
    if not (filename.exists() and filename.stat().st_size > 0):
        # Checkpoint doesn't exist.
        checkpoint: int = 0
        component_count_accumulator: List[int] = [nx.number_connected_components(G)]
    else:
        # Checkpoint exists.
        checkpoint: int
        component_count_accumulator: List[int]
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        try:
            checkpoint, component_count_accumulator, G = data
            print(f'<FMORN> [{checkpoint}/{len(groups)}] Checkpoint found!')
        except ValueError:
            # Assume routine previously completed successfully.
            print('<FMORN> Routine previously completed successfully, returning result.')
            component_count_accumulator, G = data
            return component_count_accumulator, G
    # Define flattening function.
    def flatten_group(group: Set[Vertex]):
        if not group or len(group) == 1:
            raise ValueError('B R U H')
        group_iter = iter(group)
        vertex: Vertex = next(group_iter)
        to_delete: List[Vertex] = list(group_iter)
        neighbourhood: Set[Vertex] = set()
        neighbourhood.update(*(set(G.neighbors(v)) for v in to_delete))
        G.add_edges_from((vertex, neighbour) for neighbour in neighbourhood)
        G.remove_nodes_from(to_delete)
        for node in to_delete:
            del G.graph['pos'][node]
    # Flatten groups, keeping track of number of connected components.
    print(f'Flattening {pluralise(len(groups), "group")}...')
    start: float = time.perf_counter()
    for i, group in enumerate(groups[checkpoint:]):
        debug: bool = i and not i % every
        if debug:
            now: float = time.perf_counter()
            print(f'[{i}/{len(groups)}] Flattened {pluralise(every, "group")} in {sec2str(now-start)}...' + ' '*30 + '\r', end='')
            with open(filename, 'wb') as handle:
                pickle.dump((i, component_count_accumulator, G), handle)
            start = now
        flatten_group(group)
        component_count_accumulator.append(nx.number_connected_components(G))
    print('\nAll overlapping groups of vertices flattened.')
    with open(filename, 'wb') as handle:
        pickle.dump((component_count_accumulator, G), handle)
    print(f'<FMORN> Result saved ({humanize.naturalsize(filename.stat().st_size)}).')
    return component_count_accumulator, G

def attempt_to_resolve_MORN_using_cncv(G: nx.Graph=None, print_every: int=100) -> nx.Graph:
    '''
        MORN resolution strategy:
            1) Find the groups of overlapping vertices.
            2) Merge all neighbours in the group to a single member.
            3) Delete all other members of the group.
            4) Repeat 1-3 until all groups addressed.
            5) Count number of components.
    '''
    # Obtain G.
    if G is None:
        G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    # Obtain flattened MORN.
    component_count_accumulator: List[int]
    component_count_accumulator, G = flatten_MORN_using_cncv(G)
    # Plot data.
    plt.plot(range(len(component_count_accumulator)), component_count_accumulator)
    # Show data.
    plt.show()
    # counts, edges, bars = plt.hist(sizes)
    # plt.bar_label(bars)
    # plt.show()

def distances_covered_by_local_cutvertices_MORN(checkpoint_file: Path, save_every: int=100) -> Dict[Vertex, float]:
    '''
        For each pickled local cutvertex v with locality r, this function computes
        and saves the distance covered by the ball of radius r/2 around v.

        Parameters
        ----------
        checkpoint_file: Path
            The checkpoint file for resuming and incrementally saving progress.
        save_every: int, default 100
            In between how many processed local cutvertices should we be saving our progress?

        Notes
        -----
        As the units of the MORN dataset's vertices' coordinates is currently
        unknown to me, I'll just use Euclidean distance for now and convert to
        miles or kilometers when appropriate instead of wasting my time trying
        to figure that out now.

        Posterior note: the units are meters.

        Returns
        -------
        Dict[Vertex, float]
            A dictionary of the distances covered by the aforedescribed balls and
            their respective vertices.
    '''
    # Get G
    G: nx.Graph = __get_MORN()
    # Get the coordinates.
    pos = G.graph['pos']
    # Get the pickled local cutvertices.
    local_cutvertices: List[LocalCutvertex] = __get_pickled_local_cutvertices(G)
    # Define the function that computes the edge length.
    def compute_edge_length(edge: Tuple[Vertex, Vertex]) -> float:
        x, y = edge
        (a, b), (c, d) = pos[x], pos[y]
        return math.sqrt((a-c)**2 + (b-d)**2)
    # Check for checkpoint file.
    if not (checkpoint_file.exists() and checkpoint_file.stat().st_size > 0):
        # Checkpoint doesn't exist, start from scratch.
        checkpoint: int = 0
        distances_covered: Dict[Vertex, float] = dict()
    else:
        # Checkpoint found, attempt loading data.
        checkpoint: int
        distances_covered: Dict[Vertex, float]
        with open(checkpoint_file, 'rb') as handle:
            data = pickle.load(handle)
        try:
            checkpoint, distances_covered = data
            print(f'<DCBLC-MORN> [{checkpoint}/{len(local_cutvertices)}] Checkpoint found!')
        except ValueError:
            # Assume routine previously ran to completion.
            print('<DCBLC-MORN> Routine previously ran to completion, returning result...')
            distances_covered = data
            return distances_covered
    # Proceed with routine.
    start: float = time.perf_counter()
    for i, lcv in enumerate(local_cutvertices[checkpoint:]):
        # Should we save our progress?
        if i and not i % save_every:
            now: float = time.perf_counter()
            # Save progress.
            with open(checkpoint_file, 'wb') as handle:
                pickle.dump((i, distances_covered), handle)
            print(f'<DCBLC-MORN> [{checkpoint + i}/{len(local_cutvertices)}] Processed {save_every} local cutvertices in {sec2str(now-start)}' + ' '*60 + '\r', end='')
            start = now
        # Obtain ball around local cutvertex.
        ball_around_lcv: nx.Graph = ball(G, lcv.vertex, lcv.locality / 2)
        # Obtain lengths of each edge.
        lengths = map(compute_edge_length, ball_around_lcv.edges())
        # Sum all the lengths.
        distance_covered: float = sum(lengths)
        # Add to dictionary.
        distances_covered[lcv.vertex] = distance_covered
    # Routine complete, save all progress.
    print('\n<DCBLC-MORN> Routine complete! Saving overall progress...')
    with open(checkpoint_file, 'wb') as handle:
        pickle.dump(distances_covered, handle)
    print(f'<DCBLC-MORN> Progress saved ({humanize.naturalsize(checkpoint_file.stat().st_size)}). Returning result...')
    return distances_covered

def plot_distances_covered_radii_MORN():
    # Get the graph.
    G: nx.Graph = __get_MORN()
    # Get the local cutvertices.
    local_cutvertices: List[LocalCutvertex] = __get_pickled_local_cutvertices(G)
    # Get the distances covered.
    checkpoint_file: Path = PROJECT_ROOT / 'MORN' / 'DistancesCoveredByLocalCutvertexBalls.pickle'
    distances: Dict[Vertex, float] = distances_covered_by_local_cutvertices_MORN(
        checkpoint_file
    )
    # Construct scatter plot data.
    radii: List[int] = [lcv.locality for lcv in local_cutvertices]
    distances: List[float] = [distances[lcv.vertex] for lcv in local_cutvertices]
    # Scatter plot.
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    # Format x-axis.
    xlabels = [
        3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 150, 250, 500, 1000, 1250, 1750,
    ]
    ax.set_xticks(xlabels)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    threshold_line_params = {
        'linewidth': .5,
        'color': 'black',
        'linestyle': 'dashed',
        'alpha': 0.1
    }
    for xlabel in xlabels:
        ax.axvline(x=xlabel, **threshold_line_params)
    # Format y-axis.
    ylabels = [
        1, 10, 50, 100, 1_000, 5_000, 10_000, 25_000, 50_000,
        100_000, 300_000, 410_000,
    ]
    ax.set_yticks(ylabels)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for ylabel in ylabels:
        ax.axhline(y=ylabel, **threshold_line_params)
    # Plot.
    ax.scatter(radii, distances, s=1)
    ax.set_xlabel('radius $r$ of local cutvertex $v$')
    ax.set_ylabel(r'meters of road covered by $B_\frac{r}{2}(v)-v$')
    # Try fitting line.
    logr = np.log(radii)
    logd = np.log(distances)
    coeffs = np.polyfit(logr, logd, deg=1)
    poly = np.poly1d(coeffs)
    yfit = lambda x: np.exp(poly(np.log(x)))
    ax.plot(
        np.array(radii),
        yfit(np.array(radii)),
        'r-',
    )
    # Show.
    print(coeffs)
    plt.show()

def large_plot_local_cutvertices_by_distance_covered_MORN():
    # Get distances covered.
    checkpoint_file: Path = PROJECT_ROOT / 'MORN' / 'DistancesCoveredByLocalCutvertexBalls.pickle'
    distances_covered: Dict[Vertex, float] = distances_covered_by_local_cutvertices_MORN(checkpoint_file)
    items: List[Tuple[Vertex, float]] = list(distances_covered.items())
    items.sort(key=operator.itemgetter(1), reverse=True) # Sort vertices by descending order of the area their balls cover (HA!)
    # Plot some of the best few.
    G: nx.Graph = __get_MORN()
    local_cutvertices: List[LocalCutvertex] = __get_pickled_local_cutvertices(G)
    best: int = 50
    filename_template: str = '{i},{vertex},{distance}.png'
    for i, (vertex, distance) in enumerate(items[:best]):
        filename: Path = PROJECT_ROOT / 'media' / 'MajorOpenRoadNetwork' / 'DistancesCovered' / filename_template.format(i=i, vertex=vertex, distance=round(distance, 2))
        if filename.exists():
            print(f'[{i+1}/{len(items[:best])}] Skipping {vertex}...')
            continue
        chosen_one: LocalCutvertex = next(lcv for lcv in local_cutvertices if lcv.vertex == vertex)
        large_plot_MORN(G, special=chosen_one, fname=path_to_str(filename), figure_size=A4, dpi=300)    
        plt.clf()
        plt.cla()
        print(f'[{i+1}/{len(items[:best])}] Processed.')

def stackoverflow_interesting_components(G: nx.Graph=None):
    if G is None:
        G: nx.Graph = stackoverflow['stackoverflow']
    with open(_pickle_name(G.name), 'rb') as handle:
        local_cutvertices: List[LocalCutvertex] = pickle.load(handle)
    lcvs: Set[Vertex] = set(lcv.vertex for lcv in local_cutvertices)
    H: nx.Graph = split_at_local_cutvertices(G, local_cutvertices)
    H.remove_nodes_from(lcvs)
    
    # Ok, now let's look at the interesting components.
    for comp in nx.connected_components(H):
        if any(H.nodes[node].get('split', False) for node in comp):
            print('Interesting component found!')
            print(comp)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> --- <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == '__main__':
    w13_every_other_pair_of_spokes_removed()
    exit()
    # MORN_SPLIT_COMPONENT_SIZES()
    # exit()

    # MORN_SPLIT_COMPONENT_ISOMORPHISM_TEST(3)
    # exit()

    k = 3
    INFPOWER_SPLIT_COMPONENT_ISOMORPHISM_TEST(k)
    exit()

    INFPOWER_SPLIT()
    exit()

    INFPOWER_PLOT(nx.kamada_kawai_layout)
    exit()

    INFPOWER_COMPARISON()
    exit()

    MORN_1739_ROAD_NAMES()
    exit()

    MORN_1739_LOCAL()
    exit()

    plot_distances_covered_radii_MORN()
    exit()

    MORN_LOCAL_CUTVERTICES_RADII()
    exit()

    large_plot_MORN(fname=None)
    exit()

    NDMJ20_NO_LOCAL_CUTVX()
    exit()

    from functools import partial
    layout = partial(nx.spring_layout, k=0.90, iterations=40)
    # layout = nx.kamada_kawai_layout
    SO_PLOT_COMMUNITIES(layout)
    exit()

    SO_LOCAL_CUTVERTICES_BALLS_latex()
    exit()

    from functools import partial
    layout = partial(nx.spring_layout, k=0.90, iterations=40)
    SO_LOCAL_CUTVERTICES_BALLS(layout, EDGE_SCALE=5.)
    exit()

    SO_LOCAL_CUTVERTICES_GROUPS_table()
    exit()

    G: nx.Graph = infpower['inf-power']
    local_cutvertex_radii_distribution(G, thresholds=[10, 25, 75, 125], weird=True)
    exit()

    SO_COMMUNITIES_table()
    exit()

    G: nx.Graph = stackoverflow['stackoverflow']
    print(nx.info(G))
    exit()
    # lcvs: List[LocalCutvertex] = __get_pickled_local_cutvertices(G)
    # for lcv in lcvs:
    #     print(lcv)
    # local_cutvertex_radii_distribution(G)
    exit()

    # from functools import partial
    # layout = partial(nx.spring_layout, k=0.70, iterations=40)
    layout = nx.kamada_kawai_layout
    SO_PLOT_COMMUNITIES(layout)
    exit()

    # NDMJS20_NO_LOCAL_CUTVX_HYPOTHESIS()
    # exit()

    # NDMJS20_NO_LOCAL_CUTVX_LEAVES()
    # exit()

    DECIMAL_PLACES: int = 4
    NDMJS20_NO_LOCAL_CUTVX_table(DECIMAL_PLACES)
    exit()


    N: int = 10
    gs = [
        ('path', nx.path_graph(N)),
        ('cycle', nx.cycle_graph(N)),
        ('complete', nx.complete_graph(N))
    ]
    for name, G in gs:
        print(name.capitalize(), 'graph on', pluralise(N, 'vertex'))
        print('L(G):', CPL(G))
        print('C(G):', CC(G))
        print('='*10)
    exit()



    G: nx.Graph = NDMJS20['rat_brain_1']
    local_cutvertex_radii_distribution(G)
    # print('playground')
    # __radii_Network_Data_MJS20()
    exit()
    # definition_splitting_local_cutvertex()
    # exit()

    fname: Path = PROJECT_ROOT.parent / 'Thesis' / 'major-road-network.png'
    large_plot_MORN(fname=fname)
    exit()

    plot_distances_covered_radii_MORN()
    exit()

    G: nx.Graph = __get_MORN()
    thresholds: bool = False
    weird: bool = False
    local_cutvertex_radii_distribution(G, thresholds=thresholds, weird=weird)
    exit()

    fname: Path = PROJECT_ROOT.parent / 'Thesis' / 'collapsing-overlapping-groups.png'
    OG_COLLAPSING_NUM_COMPONENTS_MORN(fname=fname)
    exit()

    group_size: int = 7
    seed: int = 420
    OG_MORN(group_size=group_size, seed=seed)
    exit()

    G: nx.Graph = __get_MORN()
    large_plot_MORN(G, fname=None)
    exit()
    # large_plot_MORN(fname='jamie.png', figure_size=A4, dpi=600)
    # exit()
    # look_at_cncv_MORN()
    # attempt_to_resolve_MORN_using_cncv()
    # _ = overlapping_groups_MORN()
    # _, F = flatten_MORN_using_cncv(G=G)
    # print('<main> Deleting missing vertices from flattened MORN pos dictionary...')
    # diff: Set[Vertex] = G.nodes() - F.nodes()
    # for missing in diff:
    #     del F.graph['pos'][missing]
    # print('<main> Obtaining redundant points for flattened MORN...')
    # redundant_points: List[Vertex] = redundant_points_MORN(G=F)
    # savings_ratio: float = 1 - (F.number_of_nodes() - len(redundant_points)) / F.number_of_nodes()
    # savings_percentage: float = round(savings_ratio * 100, 2)
    # print('<main> Removing redundant points from flattened MORN...')
    # remove_redundant_points(F, redundant_points)
    # print(f'<main> Trimmed, flattened MORN is {savings_percentage}% lighter than flattened MORN.')
    # print('<main> Updating shortest path lengths...')
    # F.graph['shortest_path_lengths'] = dict(nx.all_pairs_shortest_path_length(F))
    # print('<main> Updated shortest path lengths.')
    # print('<main> Finding local cutvertices in trimmed, flattened MORN...')
    # checkpoint_file: Path = PROJECT_ROOT / 'MORN' / 'trimmed_flattened_MORN_flc.pickle'
    # checkpoint_file: Path = PROJECT_ROOT / 'MORN' / 'CLEAN_POST_PROCESSED_MORN.pickle'
    # flc(G, checkpoint_file)
    # exit()
    

    component_count, G = flatten_MORN_using_cncv()
    plt.plot(range(len(component_count)), component_count, linestyle='--')
    plt.show()
    exit()
    pos = G.graph['pos']
    components: List[Set[Vertex]] = list(nx.connected_components(G))
    print(f'We have {pluralise(len(components), "connected component")} in the flattened version of MORN.')
    
    highlight: str = 'xkcd:salmon'
    no_highlight: str = 'gray'

    try:
        for component in components:
            fig, ax = plt.subplots()
            nx.draw_networkx(G, pos, nodelist=component, node_color=highlight, ax=ax)
            nx.draw_networkx(G, pos, nodelist=G.nodes()-component, node_color=no_highlight, ax=ax)
            plt.show()
    except KeyboardInterrupt:
        print('Stopping...')
    
    # definition_splitting_local_cutvertex()


    # leaves_inter_node_distance_MORN(G=G)

    # HOW_MANY: int = 25
    # random_plot_MORN_components(HOW_MANY, G=G)

    # nx.draw(G, G.graph['pos'], node_size=1, node_color='red', edge_color='gray')
    # figure_size: FigureSize = A2
    # dpi: int = 1200

    # large_plot_MORN(G=G, figure_size=figure_size, dpi=dpi)
    # compare_MORN_redundant_points()
    # how_many: int = 5
    # analyse_MORN(how_many)
    # local_cutvertex_radii_distribution(G)
    # __get_Network_Data_MJS20_graphs()

    # graph: str = 'net_AG'
    # min_locality: int = 7
    # try_draw_split_vertices(graph, min_locality=min_locality)

    # name: str = 'net_AG'
    # G: nx.Graph = NDMJS20[name]
    # local_cutvertex_radii_distribution(G)

    # number_of_components_post_splitting_NDMJS20()
    # is_this_definitely_working()

    # layout: callable = nx.kamada_kawai_layout
    # w13_every_other_pair_of_spokes_removed(layout)

    # __radii_Network_Data_MJS20()

    # try_draw_split_vertices('net_green_eggs')
    # split_vertices()
    # __radii_Network_Data_MJS20(name='howareyoudifferent')
    # NDMJS20_sorted_by_local_cutvertex_count()
    # graphs_with_no_neighbouring_local_cutvertices()
