# Learning GNNs with ENZYMES

A small experimental project to practise graph neural networks using PyTorch
Geometric and the ENZYMES dataset. The task is **graph classification**:
assigning each of its 600 protein graphs to one of six enzyme classes.

The notebooks explore Graph Convolutional Networks (GCNs) and Graph Attention
Networks (GATs), with experiments on feature standardization, model depth,
regularization, attention heads, and mean versus sum pooling. This project also
introduces training with batches of graphs.

- `notebooks/`: dataset inspection and model experiments.
- `src/models.py`: model definitions.
- `results/`: saved metrics and learning curves.
- `requirements.txt`: Python dependencies.

Run the notebooks in numerical order using the project's Python environment.
The dataset is downloaded automatically into `data/` when first loaded.

Next step: explore distance-based edge attributes with QM9, using atom
coordinates to calculate distances and models that explicitly use those edge
features. QM9 introduces regression rather than classification.
