import numpy as np
import sys
import matplotlib as plt

def WeisferLheman_kernel(G, n_iter, representation = False):
    r"""
    G is a list of graphs wich each graph is saved as:
        (g, lab), where:
        > g is a graph consisting of a dictionariy with keys, u_0,
        the nodes, that assigns a list  with  the nodes v,
        that have an edge (v,u)
        > lab is a dictionary that to each node assigns a label
    Returns:
        > Matrix Kernel
    """

    ng = len(G) # Number of grafs
    result = np.zeros((ng,ng))
    
    first_labels = set()
    idx_labels = 0
    # We iterate all the graphs to first 
    for (g, l) in G:
        first_labels |= set(l.values())
    
    L = list(first_labels)
    # A cada node li assignam lo que sirà la seva representació
    labels_count = {}
    WL_inverse_labels = {L[i]: i for i in range(len(L))}
    for k in range(len(G)):
        g, l = G[k][0], G[k][1]
        for v in g.keys(): 
            # Assignam nou label a cada node
            l[v] = WL_inverse_labels[l[v]]
    idx_labels = len(L)

    # Feim l'algoritme dek WeisfeirelLehman n_iter vegades
    for i in range(n_iter):
        labels_saved = set()
        for k in range(len(G)):
            new_labels = set()
            g, l = G[k][0], G[k][1]
            actual_node_labels = {}
            # Feim reassignació
            for v in g.keys():
                # Assignam noves credencials
                credential = str(l[v]) + ',' +\
                        str(sorted([l[u] for u in g[v]]))
                actual_node_labels[v] = credential

            # Reassignam les credencials modificades per credencials No idx
            for v in actual_node_labels.keys(): 
                l[v] = actual_node_labels[v] 

            # Miram quins labels s'han afegit que no estaven en altres grafs
            new_labels |= set(actual_node_labels.values())
            new_labels = new_labels - labels_saved

            # Assignam als nous labels un idx
            L_new = sorted(list(new_labels))
            for lab in L_new:
                WL_inverse_labels[lab] = idx_labels
                idx_labels += 1

            # Guardam aquests labels com a processats
            labels_saved |= new_labels
            
            for i in actual_node_labels.keys():
                # Afegim conteo al graf
                labels_count[k][WL_inverse_labels[actual_node_labels[i]]] = 0
                l[i] = WL_inverse_labels[actual_node_labels[i]]
            
            # Contam quants de labels hi ha repetits
            for i in actual_node_labels.keys():
                labels_count[k][l[i]] += 1

    if representation: return labels_count
    # Compute distances     
    for i in range(len(G)):
        for j in range(i,len(G)):
            k1 = set(list(labels_count[i].keys()))
            k2 = set(list(labels_count[j].keys()))

            common_labels = k1.intersection(k2)
            kk = 0
            kk1 = 0
            kk2 = 0
            for l in list(common_labels):
                kk += labels_count[i][l]*labels_count[j][l]
                kk1 += labels_count[i][l]**2
                kk2 += labels_count[j][l]**2
            kk = kk/np.sqrt(kk1*kk2)

            result[i, j], result[j, i] = kk, kk

        char = f"\033[7m \033[0m"*int(100*i/(ng)) + " "*(100-int(100*i/(ng)))
        sys.stdout.write(f"\r|{char}|")  # Escribe en la misma línea
        sys.stdout.flush()  # Forzar la actualización de la línea
    f = " "*102 
    sys.stdout.write(f"\r{f}")
    sys.stdout.write(f"\rDistance matrix computed!\n")
    sys.stdout.flush()

    return result


def graph_labeled(n,S,D,labels = dict(),grak = False):
    r"""
    This function recieves a set of edges, S (source) and D
    (destination) of a graph with n noodes. The dictionary 
    labels assigns to each node a label, if is empty we assign
    to each node the same label.
    """

    if len(labels.keys()) == 0: labels = {i: 'a' for i in range(n)}
    elif len(labels.keys())!= n: 
        print(f'WARNING: The label dictionary must be of length {n}') 
        labels = {i: 'a' for i in range(n)}
    
    graph = {i: [] for i in range(n)}
    for i in range(len(S)):
        graph[D[i]].append(S[i])

    return (graph, labels) if not grak else (graph, labels)


