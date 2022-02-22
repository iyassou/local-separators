'''
    playground.py   -   Code used to mainly test visualise.py
'''

from . import (
    _pickle_name,
)

from .datasets import Network_Data_MJS20 as NDMJS20
from .local_separators import (
    Vertex,
    split_at_vertices,
    ball,
)
from .utils import (
    seconds_to_string as sec2str,
)
from .visualise import (
    draw_graph,
    draw_local_cutvertices,
    draw_locality_heatmap,
    draw_split_vertices,
)

from collections import Counter
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

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import time

# UTILITIES

def __get_Network_Data_MJS20_graphs() -> List[nx.Graph]:
    categories: List[str] = 'FoodWebs Genetic Language Metabolic Neural Social Trade'.split()
    datasets: List[List[nx.Graph]] = list(map(lambda x: NDMJS20[x], categories))
    graphs: List[nx.Graph] = [graph for dataset in datasets for graph in dataset]
    return graphs

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

def try_draw_split_vertices(graph: str, min_radius: int=None):
    MIN_RADIUS: int = min_radius or 3
    G: nx.Graph = NDMJS20[graph]
    with open(_pickle_name(G.name), 'rb') as handle:
        local_cutvertices: Dict[Vertex, int] = pickle.load(handle)
    local_cutvertices: Dict[Vertex, int] = {
        k:v for k,v in local_cutvertices.items() if v >= MIN_RADIUS
    }
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
    local_cutvertices: List[Dict[Vertex, int]] = []
    for graph in graphs:
        with open(_pickle_name(graph.name), 'rb') as handle:
            local_cutvertices.append(pickle.load(handle))
    min_radius: int = 4
    scores = np.array(
        [
            sum(
                v for k,v in Counter(lc.values()).items()
                if k > min_radius
            )
            for lc in local_cutvertices
        ]
    )
    inds = np.argsort(scores)[::-1] # descending order
    top_n: int = len(inds)
    with open('sorted_by_local_cutvertex_count.txt', 'w') as f:
        for i, graph in enumerate(graphs[inds][:top_n]):
            idx = inds[i]
            lc: Dict[Vertex, int] = local_cutvertices[idx]
            rad_count = Counter(filter(lambda x: x > min_radius, lc.values()))
            f.write(f'Position #{i+1} (Score {scores[idx]}): {graph.name.stem}\n')
            for rad, count in rad_count.items():
                f.write(f">>> {pluralise(count, f'{rad}-local cutvertex')}\n")
            f.write('-'*20 + '\n')
    print('Done.')

# FIGURES

def __radii_Network_Data_MJS20(min_radius: int=None):
    from . import _pickle_name
    import pickle
    # Obtain all the graphs in the dataset.
    graphs: List[nx.Graph] = __get_Network_Data_MJS20_graphs()
    # Concatenate all the radii.
    if min_radius is None:
        min_radius: int = 0
    radii = []
    for graph in graphs:
        with open(_pickle_name(graph.name), 'rb') as handle:
            local_cutvertices: Dict[Vertex, int]= pickle.load(handle)
        radii.extend(
            filter(lambda x: x >= min_radius, local_cutvertices.values())
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
    plt.savefig('ndmjs20_radii_distribution.png', dpi=1200)

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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> --- <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == '__main__':
    # try_draw_split_vertices('net_green_eggs')
    split_vertices()