from datasets.graph_data_reader import GraphData as gd
import torch, math
import numpy as np
import os

A = '1, 2, 3'
print(list(map(int,A.split(sep=','))))
B = [[1,2,3],[0,0,1]]
print(np.array(B).shape)