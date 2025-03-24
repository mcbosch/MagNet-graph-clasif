from datasets.graph_data_reader import GraphData as gd
import torch, math
import numpy as np
import os

f = np.matrix([
    [1,2,1],
    [2,1,1],
    [1,1,1]
])

g2 = np.matrix([
    [1, 0.5],
    [0.5, 1]
])

g = np.array([1, 0.5, 0.5, 1])

print(np.dot(g, np.array([1,1,1,1])))