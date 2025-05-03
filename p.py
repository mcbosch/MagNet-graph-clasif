from datasets.graph_data_reader import GraphData as gd
import torch, math
import numpy as np
import os
A = np.matrix([
    [8, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 37/8, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0   , 144/37, 1, 0,0, ]
])