# gnn-learning-enzymes

## Next step after ENZYMES

Finish the ENZYMES graph classification exercises first. Then practise GNNs
with edge attributes, focusing on distances between connected nodes for the
thesis.

- Store each edge's distance in the matching row of `edge_attr`; distances alone
  have shape `[num_edges, 1]`.
- Choose which nodes connect separately from calculating their distances.
- Use a model that explicitly consumes `edge_attr` during message passing.
- Consider [QM9](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.QM9.html)
  as the next dataset: it provides atom coordinates for computing distances.
  Its prediction tasks are regression rather than classification.
