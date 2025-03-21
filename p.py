from datasets.graph_data_reader import GraphData as gd
import torch, math
import numpy as np
import os

files = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '\datasets\RGRAPH')

print(list(filter(lambda f: f.find('graph_indicator') >= 0, files)))

print(files[2].find('hola'))