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
    MajorOpenRoadNetworks,
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
    escape_underscore,
    pluralise,
    collinear,
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

def __get_W13_special(hub: int=1) -> nx.Graph:
    G: nx.Graph = nx.cycle_graph(range(hub+1, hub+13))
    spokes = []
    for i in range(hub+1, hub+13, 4):
        spokes.extend([(hub, i), (hub, i+1)])
    G.add_edges_from(spokes)
    return G

def __get_MORN() -> nx.Graph:
    return MajorOpenRoadNetworks['Major_Road_Network_2018_Open_Roads']

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
            local_cutvertices: List[LocalCutvertex]= pickle.load(handle)
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

def w13_every_other_pair_of_spokes_removed(layout: callable):
    G: nx.Graph = __get_W13_special(hub=1)
    pos = layout(G)
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    nx.draw_networkx(G, pos, with_labels=True, font_color='w', ax=ax)
    ax.set_title('Graph $H$, isomorphic to $W^{14}$ with every other pair of spokes removed')
    ax = axes[1]
    B = ball(G, 1, 1.5)
    nx.draw_networkx(B, pos, with_labels=True, font_color='w', ax=ax)
    ax.set_title('$B_{3/2}(1)$ in $H$')
    plt.show()

def definition_splitting_local_cutvertex():
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
    # Introduce some figure hyperparameters.
    NODE_SIZE: int = 250
    DEFAULT_EDGE_WIDTH: float = 2
    THICK_EDGE_WIDTH: float = 3
    FONT_WEIGHT: str = 'bold'
    FONT_COLOUR: str = 'k'
    EDGECOLORS: str = 'k'
    
    ### Construct the different graphs we'll be plotting.
    ## G is the graph on the left, plain and simple, already there.
    ## H is the graph in the middle, apart from some highlighting
    ## when being drawn, it is identical to G.
    H: nx.Graph = G
    ## I is the graph on the right, it'll be G split at hub with radius r.
    r: int = 3
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
        font_color=FONT_COLOUR,
        font_weight=FONT_WEIGHT,
        ax=ax,
    )
    # Draw H, which is G with the ball highlighted.
    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_title(rf'$G$ with $B_{{{r}/{2}}}({hub_name})$ highlighted')
    non_ball_nodes: Set[Vertex] = G.nodes() - ball_around_hub.nodes()
    non_ball_edges: List[Tuple[Vertex, Vertex]] = [
        (x,y) for (x,y) in G.edges()
        if x in non_ball_nodes or y in non_ball_nodes
    ]
    nx.draw_networkx(
        ball_around_hub, pos,
        node_size=NODE_SIZE,
        node_color=BALL_COLOUR,
        width=THICK_EDGE_WIDTH,
        edgecolors=EDGECOLORS,
        edge_color=BALL_COLOUR,
        labels=labels,
        font_color=FONT_COLOUR,
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
        font_color=FONT_COLOUR,
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
        font_color=FONT_COLOUR,
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
    # showmepls
    plt.show()

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
    # leggo
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
        component: Set[Vertex] = next(comp for comp in components if v in comp)
        ma: int = len(component)
        if mi > ma:
            continue
        mid: int = None
        v_is_a_local_cutvertex: bool = None
        while True:
            if ma - mi == 1 and v_is_a_local_cutvertex is not None:
                if not v_is_a_local_cutvertex:
                    v_is_a_local_cutvertex: bool = is_local_cutvertex(G, v, mi)
                    mid: int = mi
                    break
                else:
                    v_is_a_local_cutvertex: bool = is_local_cutvertex(G, v, ma)
                    if v_is_a_local_cutvertex:
                        mid: int = ma
                    else:
                        v_is_a_local_cutvertex: bool = True
                    break
            else:
                mid: int = mi + math.floor((ma - mi) / 2)
            v_is_a_local_cutvertex: bool = is_local_cutvertex(G, v, mid)
            if mi == ma:
                break
            if v_is_a_local_cutvertex:
                mi: int = mid
            else:
                ma: int = mid
        if v_is_a_local_cutvertex:
            component_without_v: nx.Graph = nx.classes.graphviews.subgraph_view(
                G, filter_node=lambda x: x in component and x != v
            )
            if nx.algorithms.components.is_connected(component_without_v):
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

def local_cutvertex_radii_distribution(G: nx.Graph):
    with open(_pickle_name(G.name), 'rb') as handle:
        local_cutvertices: List[LocalCutvertex] = pickle.load(handle)
    c = Counter(lcv.locality for lcv in local_cutvertices)
    keys, vals = c.keys(), c.values()
    total: int = sum(vals)
    fig, ax = plt.subplots(1, 1)
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
        'Local cutvertex radii distribution for '
        + escape_underscore(G.name.name)
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

def large_plot_MORN(G: nx.Graph=None, node_color: str='red', fname: str='large_plot.png', figure_size: FigureSize=A3, dpi: int=1200):
    if G is None:
        G: nx.Graph = __get_MORN()
    fig, ax = plt.gcf(), plt.gca()
    fig.set_facecolor('xkcd:pale green')
    fig.set_size_inches(*figure_size)
    fig.set_dpi(dpi)
    nx.draw(G, G.graph['pos'], node_size=0., node_color=node_color, edge_color='gray', ax=ax)
    plt.show()
    # plt.savefig(fname, dpi=dpi)

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

def look_at_how_bad_cncl_MORN_is():
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

def look_at_cncv_MORN():
    G: nx.Graph = __get_MORN()
    pos = G.graph['pos']
    closest_non_component_vertices: Dict[Vertex, Tuple[Vertex, float]] = closest_non_component_vertex_MORN(G)
    distances: List[float] = list(map(operator.itemgetter(1), closest_non_component_vertices.values()))
    zeros: int = distances.count(0)
    print('We have', pluralise(zeros, 'overlapping point'))
    exit()
    bins = [0, .1, .25, .5, .75, 1, 10, 100, 1_000, 2_000, 3_000, 5_000]
    plt.xscale('log')
    plt.yscale('log')
    print('Plotting distances...')
    start: float = time.perf_counter()
    counts, edges, bars = plt.hist(distances, bins=bins)
    end: float = time.perf_counter()
    print(f'Plotted data in {sec2str(end-start)}.')
    plt.bar_label(bars)
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
    # look_at_cncv_MORN()
    # attempt_to_resolve_MORN_using_cncv()
    # _ = overlapping_groups_MORN()
    G: nx.Graph = __get_MORN()
    _, F = flatten_MORN_using_cncv(G=G)
    print('<main> Deleting missing vertices from flattened MORN pos dictionary...')
    diff: Set[Vertex] = G.nodes() - F.nodes()
    for missing in diff:
        del F.graph['pos'][missing]
    print('<main> Obtaining redundant points for flattened MORN...')
    redundant_points: List[Vertex] = redundant_points_MORN(G=F)
    savings_ratio: float = 1 - (F.number_of_nodes() - len(redundant_points)) / F.number_of_nodes()
    savings_percentage: float = round(savings_ratio * 100, 2)
    print('<main> Removing redundant points from flattened MORN...')
    remove_redundant_points(F, redundant_points)
    print(f'<main> Trimmed, flattened MORN is {savings_percentage}% lighter than flattened MORN.')
    print('<main> Updating shortest path lengths...')
    F.graph['shortest_path_lengths'] = dict(nx.all_pairs_shortest_path_length(F))
    print('<main> Updated shortest path lengths.')
    print('<main> Finding local cutvertices in trimmed, flattened MORN...')
    checkpoint_file: Path = PROJECT_ROOT / 'MORN' / 'trimmed_flattened_MORN_flc.pickle'
    flc(F, checkpoint_file)
    exit()
    

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
