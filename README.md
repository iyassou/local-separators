# Local separators in large networks

## What's this?

In graph theory, a separator is a set of vertices which, when removed, disconnect a graph into 2 or more connected components. When the separator consists of a single vertex we call it a cutvertex. In large networks it's rather unlikely that we'd find a cutvertex that separates the entire graph. However it could be the case that removing a vertex disconnects its neighbouring vertices, up to a certain radius. In their paper titled ["Local 2-separators"](https://arxiv.org/abs/2008.03032), Carmesin introduces _local separators_, a local analogue to separators.

My masters thesis consisted of identifying local cutvertices in large networks. I've analysed a few large datasets in this way using the code in this repository.

## What's inside?

`datasets` holds the datasets I used in my thesis save for `roadNetCA`. `src` is where the interesting stuff resides. `src\local_separators.py` is the work horse that underpins this. `src/playground.py` is a large complicated mess of code snippets used for generating figures and trying stuff out in general.

`masters-thesis.pdf` is the thesis I submitted, for which I received a first! 🥳

## What's next?

As things stand, I intend on contributing to the improvements I mentioned in my conclusion in my spare time. These improvements are, in no particular order:

- porting to a graph theory library that can handle larger graphs quicker
- implementing a tree-like decomposition of a graph along its local cutvertices
- adding support for large graph visualisation libraries

