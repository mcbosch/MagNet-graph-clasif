import torch

def readout_function(x, readout, batch=None, device=None, complex = False):
  r"""
  This function takes a 3-dimensional tensor (B,N,F) of node attributes
  and returns a 2-dimensonal tensor (B,F).

  THIS FUNCTION NEEDS TO BE REVISED
    > 1. HOW NEEDS TO BE THE OUTPUT TU APPLY A CROSS-ENTROPY

  """
  if complex:
    if len(x[0].size()) == 3: 
      if readout == 'complex_max':
        return torch.max(unwind(x[0],x[1]), dim=1)
      elif readout == 'complex_avg':
        return torch.mean(unwind(x[0],x[1]), dim=1)
      elif readout == 'complex_sum':
        return torch.sum(unwind(x[0],x[1]), dim=1)
      
  else:
    if len(x.size()) == 3: 
      if readout == 'max':
        return torch.max(x, dim=1)[0] # max readout
      elif readout == 'avg':
        return torch.mean(x, dim=1) # avg readout
      elif readout == 'sum':
        return torch.sum(x, dim=1)[0] # sum readout
      
    elif len(x.size()) == 2:
      batch = batch.cpu().tolist()
      readouts = []
      max_batch = max(batch)
      
      temp_b = 0
      last = 0
      for i, b in enumerate(batch):
        if b != temp_b:
          sub_x = x[last:i]
          if readout == 'max':
            readouts.append(torch.max(sub_x, dim=0)[0].squeeze()) # max readout
          elif readout == 'avg':
            readouts.append(torch.mean(sub_x, dim=0).squeeze()) # avg readout
          elif readout == 'sum':
            readouts.append(torch.sum(sub_x, dim=0).squeeze()) # sum readout
                    
          last = i
          temp_b = b
        elif b == max_batch:
          sub_x = x[last:len(batch)]
          if readout == 'max':
            readouts.append(torch.max(sub_x, dim=0)[0].squeeze()) # max readout
          elif readout == 'avg':
            readouts.append(torch.mean(sub_x, dim=0).squeeze()) # avg readout
          elif readout == 'sum':
            readouts.append(torch.sum(sub_x, dim=0).squeeze()) # sum readout
                    
          break
          
      readouts = torch.cat(readouts, dim=0)
      return readouts
  
def unwind(x_real, x_imag):
  return torch.cat((x_real, x_imag), dim=2)
