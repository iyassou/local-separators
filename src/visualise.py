'''
    visualise.py    -   Code used to render graphs using matplotlib.

    Overall structure of this file is:

        - Bunch of imports
        - Some default-value constants
        - Types and their validators
        - Abstract Style class definition
        - NodeStyle and EdgeStyle class definitions
        - Functions for determining the area of a marker
'''

# TODO:
# - settle on convention: are we passing the layout function or the positions?

from . import (
    _media_name,
    _validate_graph_name,
)
from .local_separators import (
    ball,
    split_at_local_cutvertices,
    Vertex,
    LocalCutvertex,
)
from .utils import (
    Point2d,
    FigureSize,
    A4,
    bounding_box_2d,
    escape_underscore,
    euclidean_distance,
    nearest,
    pluralise,
    visually_distinct_colours,
)

from functools import partial
from math import pi as PI
from matplotlib.colors import is_color_like
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

import attr
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import operator
import networkx as nx

DEFAULT_DATA_AXIS_FUDGE: float = 0.1
DEFAULT_INTER_NODE_DISTANCE_FRACTION: float = 0.4
DEFAULT_FIGURE_SIZE: FigureSize = A4
DEFAULT_DPI: int = 300 # 300 is an ok clarity/computation trade-off point imo

Colour = TypeVar('Colour', str, Tuple[float, float, float])
ColourLike = TypeVar('ColourLike', Colour, List[Colour])
IntLike = TypeVar('IntLike', int, List[int])
FloatLike = TypeVar('FloatLike', float, List[float])
def validate_colourlike(instance, attribute, obj: Any) -> bool:
    '''
        Notes
        -----
        is_color_like will return False if given a list, which could potentially
        be a list of nothing but Colour objects, hence the order in which we
        typecheck matters here (because of short-circuit evaluation).
    '''
    return isinstance(obj, list) and all(is_color_like(x) for x in obj) or is_color_like(obj)
def validate_native_type_like(instance, attribute, obj: Any, t: type) -> bool:
    return isinstance(obj, t) or isinstance(obj, list) and all(isinstance(x, t) for x in obj)
validate_intlike: callable = partial(validate_native_type_like, t=int)
validate_floatlike: callable = partial(validate_native_type_like, t=float)
def validate_node_shape(instance, attribute, obj: Any) -> bool:
    try:
        return obj in {'s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8'}
    except TypeError: # unhashable
        return False
def validate_edge_style(instance, attribute, obj: Any) -> bool:
    '''
        Notes
        -----
        I don't really care to write a whole proper style validator here,
        I'm honestly just seeking to be consistent.
        heh, over-engineering detected.
    '''
    if not isinstance(obj, str):
        return False
    STYLES: Dict[str, str] = {
        '-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'
    }
    return style in STYLES or style in STYLES.values()

class Style:
    def asdict(self) -> dict:
        '''
            Returns a dictionary of attr-defined attributes contained in self,
            but only those with non-None values.
        '''
        return attr.asdict(self, filter=lambda _,v: v is not None)

@attr.s(auto_attribs=True)
class NodeStyle(Style):
    '''
        node_size: IntLike, optional
            networkx default is 300.
        node_color: ColourLike, optional
            networkx default is a brighter shade of Lochmara.
        node_shape: str, optional
            networkx default is 'o', a circle.
        label: str, optional
            Label for the matplotlib legend.
        alpha: FloatLike, optional
            Node transparency: 0 highest, 1 lowest.
    '''
    node_size: IntLike = attr.ib(default=None, validator=validate_intlike)
    node_color: ColourLike = attr.ib(default=None, validator=validate_colourlike)
    node_shape: str = attr.ib(default='o', validator=validate_node_shape)
    label: str = attr.ib(default=None)
    alpha: FloatLike = attr.ib(default=None, validator=validate_floatlike)

@attr.s(auto_attribs=True)
class EdgeStyle(Style):
    '''
        edge_color: ColourLike, optional
            networkx default is 'k' (i.e. black).
        label: str, optional
            Label for the matplotlib legend.
        alpha: FloatLike, optional
            Edge transparency: 0 highest, 1, lowest.
        width: FloatLike, optional
            Line width of edges
        style: str, optional
    '''
    edge_color: ColourLike = attr.ib(default=None, validator=validate_colourlike)
    label: str = attr.ib(default=None)
    alpha: FloatLike = attr.ib(default=None, validator=validate_floatlike)
    width: FloatLike = attr.ib(default=None, validator=validate_floatlike)
    style: str = attr.ib(default=None, validator=validate_edge_style)

MarkerSizeAreaFunction = {
    'o': lambda s: PI * s * s,  # 'o' is a circle
    's': lambda s: s * s,       # 's' is a square
    '^': lambda s: 0.5 * s,     # '^' is triangle-up
}

def to_display_coordinates(ax: matplotlib.axes.Axes, point2d: Point2d, scale: float) -> Point2d:
    return tuple(
        map(
            lambda x: x * scale,
            ax.transData.transform(point2d)
        )
    )

def calculate_marker_sizes(ax: matplotlib.axes.Axes, pos: Dict[Vertex, Point2d], node_shape: str, inter_node_distance_fraction: float, method: str='min') -> Tuple[List[Vertex], List[float]]:
    '''
        Function which given a matplotlib Axes object, a dictionary of vertex positions in
        the data coordinate system, the node shape, and a method, calculates the marker size
        to use when plotting the vertices using nx.draw_networkx_nodes.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            matplotlib Axes object.
        pos: Dict[Vertex, Point2d]
            A mapping of vertices to their positions in the data coordinate system.
        node_shape: str
            The node shape to use.
        inter_node_distance_fraction: float
            Fraction of the inter-node distance to use as the radius value.
        method: str, default 'maxfit'
            The method to use for determining the node sizes. Should be either 'maxfit' or 'min'.

        Notes
        -----
        Assumes all vertices to be drawn are contained within pos.

        Returns
        -------
        Tuple[List[Vertex], List[float]]
            List[Vertex] is the list of vertices. List[float] is their respective sizes.
    '''
    # Wrangle parameters.
    if method not in {'maxfit', 'min'}:
        raise ValueError(f'unrecognised marker size calculation method "{method}"')
    if not (0 < inter_node_distance_fraction <= 1):
        raise ValueError('inter_node_distance_fraction must be a float between 0 exclusive and 1 inclusive')
    if not validate_node_shape(None, None, node_shape):
        raise ValueError(f'unrecognised marker style "{node_shape}"')
    if node_shape not in MarkerSizeAreaFunction:
        raise NotImplementedError(f'cannot automatically calculate marker size for marker style "{node_shape}"')
    # Obtain the DPI.
    DPI: float = ax.get_figure().dpi
    # Calculate the scale.
    SCALE: float = 72 / DPI     # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html#using-offset-transforms-to-create-a-shadow-effect
    # Obtain the nearest node to each node.
    closest: Dict[Vertex, Vertex] = nearest(pos)
    # Obtain the marker size area function.
    marker_size_area: callable = MarkerSizeAreaFunction[node_shape]
    # Determine the marker sizes.
    nodelist: List[Vertex] = list(pos.keys())
    ## Calculate the inter-node distances.
    distances: List[float] = [
        inter_node_distance_fraction * euclidean_distance(
            to_display_coordinates(ax, pos[node], SCALE),
            to_display_coordinates(ax, pos[closest[node]], SCALE)
        ) for node in nodelist
    ]
    ## Obtain the marker sizes depending on the method used.
    node_size: List[float]
    if method == 'min':
        min_distance: float = min(distances)
        node_size = [marker_size_area(min_distance)] * len(distances)
    else:
        node_size = list(map(marker_size_area, distances))
    # Return the result.
    return nodelist, node_size

def draw_graph(G: nx.Graph, pos: Dict[Vertex, Point2d], data_axis_fudge: float=None, inter_node_distance_fraction: float=None, method: str='min', node_style: NodeStyle=None, edge_style: EdgeStyle=None, fig_size: FigureSize=None, dpi: int=None, overwrite: bool=True, ax=None):
    '''
        Function to draw a graph.

        Parameters
        ----------
        G: nx.Graph
        pos: Dict[Vertex, Point2d]
            A mapping of vertices to their Cartesian coordinates.
        data_axis_fudge: float, optional
            Percentage expressed as a float between 0 (exclusive) and 1
            (inclusive). The bounding box around the points generated
            by layout will be expanded by this factor times the bounding
            box's height and width.
            If None, will be assigned value of DEFAULT_DATA_AXIS_FUDGE.
        inter_node_distance_fraction: float, optional
            Fraction of the inter-node distance that should be used for
            calculating the marker sizes.
            If None, will be assigned value of DEFAULT_INTER_NODE_DISTANCE_FRACTION.
        method: str, default 'min'
            The method to use when calculating the marker sizes.
        node_style: NodeStyle, optional
            If supplied, the style in which the nodes should be drawn.
            Else the default node style.
        edge_style: EdgeStyle, optional
            If supplied, the style in which the edges should be drawn.
            Else the default edge style.
        fig_size: FigureSize, optional
            Size of the figure in inches.
            If None, will be assigned value of DEFAULT_FIGURE_SIZE.
        dpi: int, optional
            Dots per inch.
            If None, will be assigned value of DEFAULT_DPI.
        overwrite: bool, default True
            Should the resulting image overwrite a potentially pre-existing
            image?
    '''
    # Validate the graph name.
    _validate_graph_name(G.name)
    # Wrangle the fudge and fraction parameters.
    if data_axis_fudge is None:
        data_axis_fudge: float = DEFAULT_DATA_AXIS_FUDGE
    if not (0 <= data_axis_fudge <= 1):
        raise ValueError(f'bounding box fudge parameter must be between 0 and 1 inclusive, got {data_axis_fudge}')
    if inter_node_distance_fraction is None:
        inter_node_distance_fraction: float = DEFAULT_INTER_NODE_DISTANCE_FRACTION
    if not (0 <= inter_node_distance_fraction <= 1):
        raise ValueError(f'fraction of inter-node distance must be between 0 and 1 inclusive, got {inter_node_distance_fraction}')
    # Wrangle the Styles.
    if node_style is None:
        node_style: NodeStyle = NodeStyle()
    if node_style.node_shape not in MarkerSizeAreaFunction:
        raise NotImplementedError('cannot calculate area of custom marker, use one of: "o", "s", "^"')
    if edge_style is None:
        edge_style: EdgeStyle = EdgeStyle(
            alpha=0.1
        )
    # Wrangle the figure size.
    if fig_size is None:
        fig_size: FigureSize = DEFAULT_FIGURE_SIZE
    # Wrangle the DPI.
    if dpi is None:
        dpi: int = DEFAULT_DPI
    # Obtain the corresponding media folder name.
    graph_media_folder: Path = _media_name(G.name)
    # Create the media folder if it doesn't exist.
    graph_media_folder.mkdir(parents=True, exist_ok=True)
    # Create the PNG filename.
    png: Path = graph_media_folder / G.name.with_suffix('.png').name
    # If we're not overwriting then return.
    if not overwrite and png.exists():
        return
    # Configure the matplotlib figure and axes.
    if ax is None:
        fig, ax = plt.gcf(), plt.gca()
    else:
        fig = ax.get_figure()
    fig.set_size_inches(*fig_size)
    fig.set_dpi(dpi)
    # Obtain the bounding box.
    bbox = bounding_box_2d(list(pos.values()), fudge=data_axis_fudge)
    (min_x, max_y), (max_x, min_y) = bbox.top_left, bbox.bottom_right
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    # Obtain the nearest node to each node.
    closest: Dict[Vertex, Vertex] = nearest(pos)
    # Obtain the marker sizes.
    nodelist, node_size = calculate_marker_sizes(
        ax, pos, node_style.node_shape, inter_node_distance_fraction, method=method
    )
    # Draw the nodes.
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_size, ax=ax, **node_style.asdict())
    # Draw the edges.
    nx.draw_networkx_edges(G, pos, ax=ax, **edge_style.asdict())
    # Sort out the legend.
    if node_style.label is not None or edge_style.label is not None:
        plt.legend(scatterpoints=1, loc='best')
    # Sort out the title.
    n_vertices: int = G.number_of_nodes()
    n_edges: int = G.number_of_edges()
    n_components: int = nx.algorithms.components.number_connected_components(G)
    name: str = '/'.join(map(escape_underscore, (G.name.resolve().parent.name, G.name.name)))
    stats: str = ', '.join(map(pluralise, [n_vertices, n_edges, n_components], ['vertex', 'edge', 'component']))
    plt.title(f'{name} ({stats})')
    # Tighten the layout because there's only one subplot.
    plt.tight_layout()
    # Save PNG.
    # plt.savefig(png, dpi=dpi)

def draw_local_cutvertices(G: nx.Graph, pos: Dict[Vertex, Point2d], local_cutvertices: List[LocalCutvertex], data_axis_fudge: float=None, inter_node_distance_fraction: float=None, local_cutvertex_style: NodeStyle=None, ball_vertex_style: NodeStyle=None, node_style: NodeStyle=None, ball_edge_style: EdgeStyle=None, edge_style: EdgeStyle=None, fig_size: FigureSize=None, dpi: int=None, overwrite: bool=True):
    '''
        Draws the supplied local cutvertices on G.

        Parameters
        ----------
        G: nx.Graph
        pos: Dict[Vertex, Point2d]
            A mapping of vertices to their Cartesian coordinates.
        local_cutvertices: List[LocalCutvertex]
            A list of local cutvertices.
        data_axis_fudge: float, optional
            Percentage expressed as a float between 0 (exclusive) and 1
            (inclusive). The bounding box around the points generated
            by layout will be expanded by this factor times the bounding
            box's height and width.
            If None, will be assigned value of DEFAULT_DATA_AXIS_FUDGE.
        inter_node_distance_fraction: float, optional
            Fraction of the inter-node distance that should be used for
            calculating the marker sizes.
            If None, will be assigned value of DEFAULT_INTER_NODE_DISTANCE_FRACTION.
        local_cutvertex_style: NodeStyle, optional
            If supplied, the style in which the local cutvertices should be drawn.
            Else the default local cutvertex style.
        ball_vertex_style: NodeStyle, optional
            If supplied, the style in which a ball vertex should be drawn.
            Else the default ball vertex style.
        node_style: NodeStyle, optional
            If supplied, the style in which the other vertices should be drawn.
            Else the default other vertex style.
        ball_edge_style: EdgeStyle, optional
            If supplied, the style in which the ball edges should be drawn.
        edge_style: EdgeStyle, optional
            If supplied, the style in which the other edges should be drawn.
            Else the default other edge style.
        fig_size: FigureSize, optional
            Size of the figure in inches.
            If None, will be assigned value of DEFAULT_FIGURE_SIZE.
        dpi: int, optional
            Dots per inch.
            If None, will be assigned value of DEFAULT_DPI.
        overwrite: bool, default True
            Should the resulting image overwrite a potentially pre-existing
            image?
    '''
    # Validate the graph name.
    _validate_graph_name(G.name)
    # Wrangle the fudge and fraction parameters.
    if data_axis_fudge is None:
        data_axis_fudge: float = DEFAULT_DATA_AXIS_FUDGE
    if not (0 <= data_axis_fudge <= 1):
        raise ValueError(f'bounding box fudge parameter must be between 0 and 1 inclusive, got {data_axis_fudge}')
    if inter_node_distance_fraction is None:
        inter_node_distance_fraction: float = DEFAULT_INTER_NODE_DISTANCE_FRACTION
    if not (0 <= inter_node_distance_fraction <= 1):
        raise ValueError(f'fraction of inter-node distance must be between 0 and 1 inclusive, got {inter_node_distance_fraction}')
    # Wrangle the Styles. Note the node sizes will be decided later.
    if node_style is None:
        node_style: NodeStyle = NodeStyle(label='$V(G)\setminus V(B_{r/2}(v))$', alpha=0.2)
    if edge_style is None:
        edge_style: EdgeStyle = EdgeStyle(alpha=0.2, width=0.7)
    if local_cutvertex_style is None:
        local_cutvertex_style: NodeStyle = NodeStyle(node_color='tab:red', label='v', alpha=1.0)
    if ball_vertex_style is None:
        ball_vertex_style: NodeStyle = NodeStyle(label='$V(B_{r/2}(v))$')
    if ball_edge_style is None:
        ball_edge_style: EdgeStyle = EdgeStyle(edge_color='green', label='$E(B_{r/2}(v))$', alpha=0.7)
    if any(x.node_shape != node_style.node_shape for x in [local_cutvertex_style, ball_vertex_style]):
        raise ValueError(f'node shape parameters must be consistent: got {repr(local_cutvertex_style.node_shape)} for local cutvertices, got {repr(ball_vertex_style.node_shape)} for ball vertices, and {repr(node_style.node_shape)} for other vertices')
    if node_style.node_shape not in MarkerSizeAreaFunction: # they're all the same here
        raise NotImplementedError('cannot calculate area of custom marker, use one of: "o", "s", "^"')
    # Wrangle the figure size.
    if fig_size is None:
        fig_size: FigureSize = DEFAULT_FIGURE_SIZE
    # Wrangle the DPI.
    if dpi is None:
        dpi: int = DEFAULT_DPI
    # Obtain the corresponding media folder name.
    graph_media_folder: Path = _media_name(G.name)
    # Create the media folder if it doesn't exist.
    graph_media_folder.mkdir(parents=True, exist_ok=True)
    # Create the PNG template filename.
    png_template = 'B_{{{r}÷2}}({v}).png'
    # Obtain the bounding box and Axes coordinate system limits.
    bbox = bounding_box_2d(list(pos.values()), fudge=data_axis_fudge)
    (min_x, max_y), (max_x, min_y) = bbox.top_left, bbox.bottom_right
    # Configure the matplotlib figure and axes.
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(*fig_size)
    fig.set_dpi(dpi)
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    # Obtain the marker sizes. 
    nodelist, node_size = calculate_marker_sizes(
        ax, pos, node_style.node_shape, inter_node_distance_fraction, method='maxfit'
    )
    min_marker_size: float = max(min(node_size), 2)
    ball_node_marker_size: float = max(min_marker_size, 20)
    # Draw the local cutvertices.
    for local_cutvertex in local_cutvertices:
        # Bound variables locally.
        v: Vertex = local_cutvertex.vertex
        r: int = local_cutvertex.locality
        # Create PNG filename.
        png: Path = graph_media_folder / png_template.format(v=v, r=r)
        # If we're not overwriting then skip.
        if not overwrite and png.exists():
            continue
        # Obtain the ball of radius r/2 around v in G.
        B: nx.Graph = ball(G, v, r/2)
        # Draw the graph in three parts.
        ## Part 1: Draw the local cutvertex.
        nx.draw_networkx_nodes(
            G, pos, nodelist=[v], node_size=ball_node_marker_size, ax=ax, **local_cutvertex_style.asdict()
        )
        ## Part 2: Draw the balls around the local cutvertex, excluding the latter.
        ball_vertex_nodelist = list(B.nodes() - {v})
        ball_edge_edgelist = list(B.edges())
        other_vertex_nodelist: Set[Vertex] = set(G.nodes() - B.nodes())
        other_edge_edgelist: Set[Vertex] = set(G.edges() - B.edges())
        nx.draw_networkx_nodes(
            G, pos, nodelist=ball_vertex_nodelist, node_size=ball_node_marker_size, ax=ax, **ball_vertex_style.asdict()
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=ball_edge_edgelist, ax=ax, **ball_edge_style.asdict()
        )
        ## Part 3: Draw the other vertices.
        other_vertex_nodelist: List[Vertex] = list(other_vertex_nodelist)
        nx.draw_networkx_nodes(
            G, pos, nodelist=other_vertex_nodelist, node_size=min_marker_size, ax=ax, **node_style.asdict()
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=other_edge_edgelist, ax=ax, **edge_style.asdict()
        )
        # Sort out the legend.
        plt.legend(scatterpoints=1, loc='best')
        # Sort out the title.
        n_vertices: int = G.number_of_nodes()
        n_edges: int = G.number_of_edges()
        n_components: int = nx.algorithms.components.number_connected_components(G)
        name: str = '/'.join(map(escape_underscore, (G.name.resolve().parent.name, G.name.name)))
        stats: str = ', '.join(map(pluralise, [n_vertices, n_edges, n_components], ['vertex', 'edge', 'component']))
        plt.title(f'{r}-local cutvertex for {name} ({stats})')
        # Tighten the layout because there's only one subplot.
        plt.tight_layout()
        # Save PNG.
        plt.savefig(png, dpi=dpi)
        # plt.show()
        # Clear the figure.
        plt.clf()
        # Re-configure the matplotlib figure and axes.
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(*fig_size)
        fig.set_dpi(dpi)
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))

def draw_split_vertices(G: nx.Graph, layout: callable, local_cutvertices: List[LocalCutvertex], data_axis_fudge: float=None, inter_node_distance_fraction: float=None, local_cutvertex_style: NodeStyle=None, split_vertex_style: NodeStyle=None, component_vertex_style: NodeStyle=None, component_edge_style: EdgeStyle=None, default_edge_style: EdgeStyle=None, fig_size: FigureSize=None, dpi: int=None, overwrite: bool=True):
    '''
        Split G at its local cutvertices to obtain H, then draw G and H.
        The split vertices and the components of H without the local cutvertices of G are distinguished.

        Parameters
        ----------
        G: nx.Graph
        layout: callable
            A function which maps vertices to their Cartesian coordinates.
        local_cutvertices: List[LocalCutvertex]
            A list of local cutvertices.
        data_axis_fudge: float, optional
            Percentage expressed as a float between 0 (exclusive) and 1
            (inclusive). The bounding box around the points generated
            by layout will be expanded by this factor times the bounding
            box's height and width.
            If None, will be assigned value of DEFAULT_DATA_AXIS_FUDGE.
        inter_node_distance_fraction: float, optional
            Fraction of the inter-node distance that should be used for
            calculating the marker sizes.
            If None, will be assigned value of DEFAULT_INTER_NODE_DISTANCE_FRACTION.
        local_cutvertex_style: NodeStyle, optional
            If supplied, the style in which the local cutvertices should be drawn.
            Else the default local cutvertex style.
        split_vertex_style: NodeStyle, optional
            If supplied, the style in which the split vertices should be drawn.
            Else the default split vertex style.
        component_vertex_style: NodeStyle, optional
            If supplied, the style in which vertices in a component of (H - local_cutvertices)
            should be drawn.
            Else the default component vertex style.
        component_edge_style: EdgeStyle, optional
            If supplied, the style in which edges in a component of (H - local_cutvertices)
            should be drawn.
            Else the default component edge style.
        default_edge_style: EdgeStyle, optional
            If supplied, the style in which edges not contained in any component of
            (H - local_cutvertices) should be drawn.
        fig_size: FigureSize, optional
            Size of the figure in inches.
            If None, will be assigned value of DEFAULT_FIGURE_SIZE.
        dpi: int, optional
            Dots per inch.
            If None, will be assigned value of DEFAULT_DPI.
        overwrite: bool, default True
            Should the resulting image overwrite a potentially pre-existing
            image?

        Notes
        -----
        First we split G at the supplied local cutvertices to obtain H.
        Secondly we find k, the number of components of H \ local_cutvertices.
        Thirdly we generate k + wiggle_room visually distinct colours.
        Fourthly we plot G with each of the k components of (H \ local_cutvertices) coloured distinctly.
        Fifthly we plot H with each of the k components of (H \ local_cutvertices), the split vertices,
        and edges between components of (H \ local_cutvertices) coloured distinctly.
    '''
    # Validate the graph name.
    _validate_graph_name(G.name)
    # Wrangle the fudge and fraction parameters.
    if data_axis_fudge is None:
        data_axis_fudge: float = DEFAULT_DATA_AXIS_FUDGE
    if not (0 <= data_axis_fudge <= 1):
        raise ValueError(f'bounding box fudge parameter must be between 0 and 1 inclusive, got {data_axis_fudge}')
    if inter_node_distance_fraction is None:
        inter_node_distance_fraction: float = DEFAULT_INTER_NODE_DISTANCE_FRACTION
    if not (0 <= inter_node_distance_fraction <= 1):
        raise ValueError(f'fraction of inter-node distance must be between 0 and 1 inclusive, got {inter_node_distance_fraction}')
    # Wrangle the Styles. Note the node sizes will be decided later.
    if split_vertex_style is None:
        split_vertex_style: NodeStyle = NodeStyle(label='split vertex', alpha=0.7)
    if local_cutvertex_style is None:
        local_cutvertex_style: NodeStyle = NodeStyle(label='local cutvertex', alpha=0.8)
    if component_vertex_style is None:
        component_vertex_style: NodeStyle = NodeStyle(label='component vertex', alpha=0.5)
    if component_edge_style is None:
        component_edge_style: EdgeStyle = EdgeStyle(label='component edge', alpha=0.8, width=1)
    if default_edge_style is None:
        default_edge_style: EdgeStyle = EdgeStyle(label='default edge', alpha=0.5, width=0.2)
    if component_vertex_style.node_shape != split_vertex_style.node_shape:
        raise ValueError(f'node shape parameters must be consistent: got {component_vertex_style.node_shape!r} for component vertices and {split_vertex_style.node_shape!r} for split vertices')
    if component_vertex_style.node_shape not in MarkerSizeAreaFunction: # they're all the same here
        raise NotImplementedError('cannot calculate area of custom marker, use one of: "o", "s", "^"')
    # Wrangle the figure size.
    if fig_size is None:
        fig_size: FigureSize = DEFAULT_FIGURE_SIZE
    # Wrangle the DPI.
    if dpi is None:
        dpi: int = DEFAULT_DPI
    # Obtain the corresponding media folder name.
    graph_media_folder: Path = _media_name(G.name)
    # Create the media folder if it doesn't exist.
    graph_media_folder.mkdir(parents=True, exist_ok=True)
    # Create the PNG filename.
    png: Path = graph_media_folder / f"SPLIT-{G.name.with_suffix('.png').name}"
    # If we're not overwriting then stop.
    if not overwrite and png.exists():
        return

    # Split at the local cutvertices to obtain H.
    H: nx.Graph = split_at_local_cutvertices(G, local_cutvertices, inplace=False)
    # Find k, the number of components in (H - local_cutvertices).
    ## Construct a set of the local cutvertices.
    lcv_set: Set[Vertex] = set(lcv.vertex for lcv in local_cutvertices)
    ## Obtain k.
    components: List[Set[Vertex]] = list(
        nx.connected_components(
            nx.classes.graphviews.subgraph_view(
                H, filter_node=lambda x: x not in lcv_set
            )
        )
    )
    k: int = len(components)
    print('k:', k)
    # We need k colours for the components, one for split vertices,
    # one for local cutvertices, and one for edges that aren't in
    # any component.
    wiggle_room: int = 3
    vd_colours: List[str] = visually_distinct_colours(k + wiggle_room)

    # Obtain the layouts.
    pos_G : Dict[Vertex, Point2d] = layout(G)
    pos_H : Dict[Vertex, Point2d] = layout(H)
    # Configure the matplotlib figure and axes.
    ## Obtain the subplots.
    fig, axes = plt.subplots(1, 2)
    ## Set figure parameters.
    fig_size: Tuple[int, int] = (fig_size[0] * 2, fig_size[1])
    fig.set_size_inches(*fig_size)
    fig.set_dpi(dpi)
    ## For each Axes, calculate its limits and the marker sizes.
    node_size_G = []
    node_size_H = []
    for i, pos in enumerate((pos_G, pos_H)):
        ### Obtain the bounding box.
        bbox = bounding_box_2d(list(pos.values()), fudge=data_axis_fudge)
        (min_x, max_y), (max_x, min_y) = bbox.top_left, bbox.bottom_right
        ### Set the Axes limits.
        axes[i].set_xlim((min_x, max_x))
        axes[i].set_ylim((min_y, max_y))
        ### Calculate the marker sizes.
        _, (node_size_G if not i else node_size_H)[:] = calculate_marker_sizes(
            axes[i], pos, component_vertex_style.node_shape, inter_node_distance_fraction, method='maxfit'
        )
    # Decide the various marker sizes for G and H.
    min_node_size_G: float = min(node_size_G)
    component_vertex_size_G: float = max( min_node_size_G, 1 )
    local_cutvertex_size_G: float = max( min_node_size_G, 10 )
    min_node_size_H: float = min(node_size_H)
    component_vertex_size_H: float = max( min_node_size_H, 1 )
    split_vertex_size: float = max( min_node_size_H, 5 )
    local_cutvertex_size_H: float = max( min_node_size_H, 10 )

    # Obtain a set of all the edges in each graph.
    edges_G: Set[Tuple[Vertex, Vertex]] = set(G.edges)
    edges_H: Set[Tuple[Vertex, Vertex]] = set(H.edges)
    # Go through each component.
    for component, colour in zip(components, vd_colours[wiggle_room:]):
        ### Plot the component's vertices.
        nx.draw_networkx_nodes(
            G, pos_G, nodelist=list(filter(G.has_node, component)),
            node_color=colour, **component_vertex_style.asdict(),
            node_size=component_vertex_size_G,
            ax=axes[0]
        )
        nx.draw_networkx_nodes(
            H, pos_H, nodelist=component, node_color=colour,
            **component_vertex_style.asdict(),
            node_size=component_vertex_size_H,
            ax=axes[1]
        )
        ### Plot the component's edges.
        edgelist: Set[Tuple[Vertex, Vertex]] = set(
            nx.classes.graphviews.subgraph_view(
                G, filter_node=lambda x: x in component
            ).edges
        )
        nx.draw_networkx_edges(
            G, pos_G, edgelist=edgelist, edge_color=colour,
            **component_edge_style.asdict(),
            ax=axes[0]
        )
        nx.draw_networkx_edges(
            H, pos_H, edgelist=edgelist, edge_color=colour,
            **component_edge_style.asdict(),
            ax=axes[1]
        )
        ### Remove the component's edges from the total sets of edges.
        edges_G.difference_update(edgelist)
        edges_H.difference_update(edgelist)
    ## Draw the secondary edges.
    # nx.draw_networkx_edges(
    #     G, pos_G, edgelist=edges_G, edge_color=vd_colours[0],
    #     **default_edge_style.asdict(),
    #     ax=axes[0]
    # )
    # nx.draw_networkx_edges(
    #     H, pos_H, edgelist=edges_H, edge_color=vd_colours[0],
    #     **default_edge_style.asdict(),
    #     ax=axes[1]
    # )
    ## Draw the split vertices.
    nx.draw_networkx_nodes(
        H, pos_H, nodelist=[v for v,yup in H.nodes(data='split') if yup],
        node_color=vd_colours[1], **split_vertex_style.asdict(),
        node_size=split_vertex_size, ax=axes[1]
    )
    ## Draw the local cutvertices.
    nx.draw_networkx_nodes(
        G, pos_G, nodelist=lcv_set, node_color=vd_colours[2],
        **local_cutvertex_style.asdict(), node_size=local_cutvertex_size_G,
        ax=axes[0]
    )
    nx.draw_networkx_nodes(
        H, pos_H, nodelist=lcv_set, node_color=vd_colours[2],
        **local_cutvertex_style.asdict(), node_size=local_cutvertex_size_H,
        ax=axes[1]
    )

    # Sort out the legends.
    for i in range(2):
        axes[i].legend(scatterpoints=1, loc='best')
    # Sort out the titles.
    name: str = '/'.join(map(escape_underscore, (G.name.resolve().parent.name, G.name.name)))
    stats = []
    for graph in (G, H):
        n_vertices: int = graph.number_of_nodes()
        n_edges: int = graph.number_of_edges()
        n_components: int = nx.algorithms.components.number_connected_components(graph)
        stats.append(
            ', '.join(map(pluralise, [n_vertices, n_edges, n_components], ['vertex', 'edge', 'component']))
        )
    axes[0].set_title(f'{name} ({stats[0]})')
    axes[1].set_title(
        f'{name} split at {pluralise(len(local_cutvertices), "local cutvertex")} ({stats[1]})'
    )
    # Tighten the layout because there's only one subplot.
    plt.tight_layout()
    # Save PNG.
    # plt.savefig(png, dpi=dpi)
    plt.show()

def draw_jan(G: nx.Graph, pos: Dict[Vertex, Point2d], local_cutvertices: Dict[Vertex, int], data_axis_fudge: float=None, inter_node_distance_fraction: float=None, split_vertex_style: NodeStyle=None, non_split_vertex_style: NodeStyle=None, split_edge_style: EdgeStyle=None, non_split_edge_style: EdgeStyle=None, fig_size: FigureSize=None, dpi: int=None, overwrite: bool=True):
    '''

        Notes
        -----
        Assumes the graph is already split at its local cutvertices in the expected, typical way, as
        described by the behaviour of the src.local_separators.split_at_local_cutvertices function.
    '''
    # Validate the graph name.
    _validate_graph_name(G.name)
    # Wrangle the fudge and fraction parameters.
    if data_axis_fudge is None:
        data_axis_fudge: float = DEFAULT_DATA_AXIS_FUDGE
    if not (0 <= data_axis_fudge <= 1):
        raise ValueError(f'bounding box fudge parameter must be between 0 and 1 inclusive, got {data_axis_fudge}')
    if inter_node_distance_fraction is None:
        inter_node_distance_fraction: float = DEFAULT_INTER_NODE_DISTANCE_FRACTION
    if not (0 <= inter_node_distance_fraction <= 1):
        raise ValueError(f'fraction of inter-node distance must be between 0 and 1 inclusive, got {inter_node_distance_fraction}')
    # Wrangle the Styles. Note the node sizes will be decided later.
    if non_split_vertex_style is None:
        non_split_vertex_style: NodeStyle = NodeStyle(label='$v\in V(G)$', alpha=0.5, node_color='tab:green')
    if non_split_edge_style is None:
        non_split_edge_style: EdgeStyle = EdgeStyle(label='$e\in E(G)$', alpha=0.6, width=0.15)
    if split_vertex_style is None:
        split_vertex_style: NodeStyle = NodeStyle(node_color='tab:purple', label='split vertex', alpha=1.0)
    if split_edge_style is None:
        split_edge_style: EdgeStyle = EdgeStyle(label='split vertex edge', edge_color='tab:orange', alpha=1.0, width=3)
    if non_split_vertex_style.node_shape != split_vertex_style.node_shape:
        raise ValueError(f'node shape parameters must be consistent: got {repr(non_split_vertex_style.node_shape)} for non-split vertices and {repr(split_vertex_style.node_shape)} for split vertices')
    if non_split_vertex_style.node_shape not in MarkerSizeAreaFunction: # they're all the same here
        raise NotImplementedError('cannot calculate area of custom marker, use one of: "o", "s", "^"')
    # Wrangle the figure size.
    if fig_size is None:
        fig_size: FigureSize = DEFAULT_FIGURE_SIZE
    # Wrangle the DPI.
    if dpi is None:
        dpi: int = DEFAULT_DPI
    # Obtain the corresponding media folder name.
    graph_media_folder: Path = _media_name(G.name)
    # Create the media folder if it doesn't exist.
    graph_media_folder.mkdir(parents=True, exist_ok=True)
    # Create the PNG filename.
    png: Path = graph_media_folder / f"SPLIT-{G.name.with_suffix('.png').name}"
    # If we're not overwriting then stop.
    if not overwrite and png.exists():
        return

    # Obtain the bounding box and Axes coordinate system limits.
    bbox = bounding_box_2d(list(pos.values()), fudge=data_axis_fudge)
    (min_x, max_y), (max_x, min_y) = bbox.top_left, bbox.bottom_right
    # Configure the matplotlib figure and axes.
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(*fig_size)
    fig.set_dpi(dpi)
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    # Obtain the marker sizes.
    nodelist, node_size = calculate_marker_sizes(
        ax, pos, node_style.node_shape, inter_node_distance_fraction, method='maxfit'
    )
    ## Draw vertices.
    min_marker_size: float = max(min(node_size), 1)
    split_vertex_marker_size: float = max(min_marker_size, 10)
    ### BLACK NOTEBOOK 2022/02/03

def draw_locality_heatmap(G: nx.Graph, pos: Dict[Vertex, Point2d], local_cutvertices: Dict[Vertex, int], cmap: Union[matplotlib.colors.LinearSegmentedColormap, str]=None, data_axis_fudge: float=None, inter_node_distance_fraction: float=None, method: str='maxfit', node_style: NodeStyle=None, edge_style: EdgeStyle=None, fig_size: FigureSize=None, dpi: int=None, overwrite: bool=True):
    '''
        Draws a locality heatmap.

        Parameters
        ----------
        G: nx.Graph
        pos: Dict[Vertex, Point2d]
            A mapping of vertices to their Cartesian coordinates.
        local_cutvertices: Dict[Vertex, int]
            A mapping of local cutvertices to their respective radii.
        cmap: Union[matplotlib.colors.LinearSegmentedColormap, str], optional
            If None, will be assigned value of plt.cm.jet.
        data_axis_fudge: float, optional
            Percentage expressed as a float between 0 (exclusive) and 1
            (inclusive). The bounding box around the points generated
            by layout will be expanded by this factor times the bounding
            box's height and width.
            If None, will be assigned value of DEFAULT_DATA_AXIS_FUDGE.
        method: str, default 'maxfit'
            The method to use when calculating the marker sizes.
        inter_node_distance_fraction: float, optional
            Fraction of the inter-node distance that should be used for
            calculating the marker sizes.
            If None, will be assigned value of DEFAULT_INTER_NODE_DISTANCE_FRACTION.
        node_style: NodeStyle, optional
            If supplied, the style in which the nodes should be drawn.
            Else the default node style.
        edge_style: EdgeStyle, optional
            If supplied, the style in which the edges should be drawn.
            Else the default edge style.
        fig_size: FigureSize, optional
            Size of the figure in inches.
            If None, will be assigned value of DEFAULT_FIGURE_SIZE.
        dpi: int, optional
            Dots per inch.
            If None, will be assigned value of DEFAULT_DPI.
        overwrite: bool, default True
            Should the resulting image overwrite a potentially pre-existing
            image?
    '''
    import time
    from .utils import seconds_to_string
    # Validate the graph name.
    _validate_graph_name(G.name)
    # Wrangle the fudge and fraction parameters.
    if data_axis_fudge is None:
        data_axis_fudge: float = DEFAULT_DATA_AXIS_FUDGE
    if not (0 <= data_axis_fudge <= 1):
        raise ValueError(f'bounding box fudge parameter must be between 0 and 1 inclusive, got {data_axis_fudge}')
    if inter_node_distance_fraction is None:
        inter_node_distance_fraction: float = DEFAULT_INTER_NODE_DISTANCE_FRACTION
    if not (0 <= inter_node_distance_fraction <= 1):
        raise ValueError(f'fraction of inter-node distance must be between 0 and 1 inclusive, got {inter_node_distance_fraction}')
    # Wrangle the Styles.
    if node_style is None:
        node_style: NodeStyle = NodeStyle()
    if node_style.node_shape not in MarkerSizeAreaFunction:
        raise NotImplementedError('cannot calculate area of custom marker, use one of: "o", "s", "^"')
    if edge_style is None:
        edge_style: EdgeStyle = EdgeStyle(
            alpha=0.05,
            edge_color='lightgrey',
        )
    # Wrangle the figure size.
    if fig_size is None:
        fig_size: FigureSize = DEFAULT_FIGURE_SIZE
    # Wrangle the DPI.
    if dpi is None:
        dpi: int = DEFAULT_DPI
    # Wrangle the colour map.
    if cmap is None:
        cmap = 'hot'
    # Obtain the corresponding media folder name.
    graph_media_folder: Path = _media_name(G.name)
    # Create the media folder if it doesn't exist.
    graph_media_folder.mkdir(parents=True, exist_ok=True)
    # Create the PNG filename.
    png: Path = graph_media_folder / f"LOCALITY_HEATMAP-{G.name.with_suffix('.png').name}"
    # If we're not overwriting then return.
    if not overwrite and png.exists():
        return
    # Configure the matplotlib figure and axes.
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(*fig_size)
    fig.set_dpi(dpi)
    # Obtain the bounding box.
    bitchimoutthecoupe = time.perf_counter()
    bbox = bounding_box_2d(list(pos.values()), fudge=data_axis_fudge)
    bitchimoutthecoupe = time.perf_counter() - bitchimoutthecoupe
    print('bounding_box_2d took:', seconds_to_string(bitchimoutthecoupe))
    (min_x, max_y), (max_x, min_y) = bbox.top_left, bbox.bottom_right
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    # Obtain the nearest node to each node.
    bitchimoutthecoupe = time.perf_counter()
    closest: Dict[Vertex, Vertex] = nearest(pos)
    bitchimoutthecoupe = time.perf_counter() - bitchimoutthecoupe
    print('nearest took:', seconds_to_string(bitchimoutthecoupe))
    # Obtain the marker sizes.
    bitchimoutthecoupe = time.perf_counter()
    nodelist, node_size = calculate_marker_sizes(
        ax, pos, node_style.node_shape, inter_node_distance_fraction, method=method
    )
    bitchimoutthecoupe = time.perf_counter() - bitchimoutthecoupe
    print('calculate_marker_sizes took:', seconds_to_string(bitchimoutthecoupe))
    # Obtain the radii.
    colors: List[int] = list(local_cutvertices.get(node, 0) for node in G.nodes())
    # Draw the nodes, capture the result to display the colorbar.
    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_size, ax=ax, node_color=colors, cmap=cmap, **node_style.asdict())
    # Draw the edges.
    nx.draw_networkx_edges(G, pos, ax=ax, **edge_style.asdict())
    # Plot the colorbar.
    plt.colorbar(nc)
    # Sort out the legend.
    if node_style.label is not None or edge_style.label is not None:
        plt.legend(scatterpoints=1, loc='best')
    # Sort out the title.
    n_vertices: int = G.number_of_nodes()
    n_edges: int = G.number_of_edges()
    n_components: int = nx.algorithms.components.number_connected_components(G)
    name: str = '/'.join(map(escape_underscore, (G.name.resolve().parent.name, G.name.name)))
    stats: str = ', '.join(map(pluralise, [n_vertices, n_edges, n_components], ['vertex', 'edge', 'component']))
    plt.title(f'{name} locality heatmap ({stats})')
    # Tighten the layout because there's only one subplot.
    plt.tight_layout()
    # Save PNG.
    plt.savefig(png, dpi=dpi)
