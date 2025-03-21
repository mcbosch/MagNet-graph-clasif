import os
import csv
import torch.nn as nn
import torch

def create_directory(directory_path):
  try:
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
  except OSError:
    print('Creating', directory_path, 'directory error')
      
def save_result_csv(file_path, result, print_column_name, mode='a'):
  with open(file_path, mode, newline='\n') as file:
    writer = csv.writer(file)
    if print_column_name:
      writer.writerow(['dataset', 'readout', 'fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5', 'fold 6', 'fold 7', 'fold 8', 'fold 9', 'fold 10', 'mean of acc', 'std of acc', 'time per epochs'])
    writer.writerow(result)

  
class PolCheb():
  r"""
  This class is defined to work with polinomials 
  of chebyshev, evaluated on tensors, since the node features 
  are a 3 dimensional tensor. 
  """
  def __init__(self, K, S, l):
      r"""
      :param: S -> is a 3-dimensional tensor that consist of a group 
                  of graph shift operators.

      We should define a tensor consisting of a spectral weight l = l_max/2 \in [0,1]
      But note that this is one for each graph. Since we add
      zeros on a matrix to add dimensions and define a 3 dimensional
      tensor, we should calculate this value before creating the batch.
      We don't know l_max, but maybe we can found a more precise (cota)
      than 2. 
      """
      self.order = K
      self.poly = []

  def pol_evaluates(self, S, x, l=1):
    r"""
    :param: S -> graphs shift operators (3-D tensor)
    :param: x -> node features (3-D tensor)
    :param: l -> spectral scale (1-D tensor)
    """