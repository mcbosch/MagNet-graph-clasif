# MagNet for Graph Clasification

This repository is design of a Graph Neural Network for the classification of graphs. This GNN is aimed to work with directed graphs, but also works with undirected graphs. 
In order to design this GNN, we used the Magnetic Laplacian since in node-levels and edge-level performed well, see [MAGNET](https://arxiv.org/pdf/2102.11391) for more detailed 
information. We used a similar idea, but changing the readout. THIS CODE IS NOT FINISHED YET.

Future objectives: 
-   Change `GraphData` object in order to add the real and imag part od a matrix.
-   Add more complex pooling layers and not the *max, avg* or *sum*.
-   Add the MagNet on a graph-level task to PyTorch

# References

-  MagNet: [arXiv document](https://arxiv.org/pdf/2102.11391)
-  Code: [gitHub repository](https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification)
