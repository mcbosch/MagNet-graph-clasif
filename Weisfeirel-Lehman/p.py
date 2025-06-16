import WeisfeirelLehman as wl

# Ejemplo de grafos
g1 = ({0:[1,3], 1:[0,2], 2:[1,3],3:[0,2,4],4:[3]}, {0:'a',1:'a',2:'a',3:'a',4:'a'})
g2 = ({0:[], 1:[0], 2:[1]}, {0:'a',1:'a',2:'a'})
graphs = [g1,g2]

# Weisfeiler-Lehman sin kernel base
w_repr = wl.WeisferLheman(graphs, n_iter=3)
print(w_repr)

# fit_transform devuelve una matriz (n_grafos x n_características)




