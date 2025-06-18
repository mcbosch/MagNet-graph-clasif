import numpy as np
import sys
import grakel

def WeisferLheman(G, n_iter):
    r"""
    G is a list of graphs wich each graph is saved as:
        (g, lab), where:
        > g is a graph consisting of a dictionariy with keys, u_0,
        the nodes, that assigns a list  with  the nodes v,
        that have an edge (v,u)
        > lab is a dictionary that to each node assigns a label
    """

    ng = len(G) # Number of grafs
    first_labels = set()
    idx_labels = 0
    # We iterate all the graphs to first 
    for (g, l) in G:
        first_labels |= set(l.values())
    
    L = list(first_labels)
    idx_labels += len(L)
    # A cada node li assignam lo que sirà la seva representació
    labels_count = {k: {i: 0 for i in range(len(L))} for k in range(len(G))}
    WL_inverse_labels = {L[i]: i for i in range(len(L))}

    for k in range(len(G)):
        g, l = G[k][0], G[k][1]
        for v in g.keys(): 
            # Assignam nou label a cada node
            l[v] = WL_inverse_labels[l[v]]
            # Sumam 1 a cada label
            labels_count[k][l[v]]+=1

    # Feim l'algoritme dek WeisfeirelLehman n_iter vegades
    for i in range(n_iter):
        new_labels = set()
        for k in range(len(G)):
            g,l = G[k][0], G[k][1]
            actual_node_labels = {}
            # Feim reassignació
            for v in g.keys():
                # Assignam noves credencials
                credential = str(l[v]) + ',' +\
                        str(sorted([l[u] for u in g[v]]))
                actual_node_labels[v] = credential


            # Reassignam les credencials modificades (si un node no ha estat mod no el consideram)
            for v in actual_node_labels.keys(): l[v] = actual_node_labels[v] 
            new_labels |= set(actual_node_labels.values())
                
        # Assignam de manera injectiva els nous labels
        L = sorted(list(new_labels))
        for i in range(len(L)):
            WL_inverse_labels[L[i]] = idx_labels + i
        for k in range(len(G)):
            for i in range(len(L)):
                labels_count[k][WL_inverse_labels[L[i]]] = 0
        idx_labels += len(L)

        # Canviam els labels per l'assignació que hem fet i contam labels
        for k in range(len(G)):
            g, l= G[k][0], G[k][1]
            for v in g.keys(): 
                # Assignam nou label a cada node
                l[v] = WL_inverse_labels[l[v]]
                # Sumam 1 a cada label
                labels_count[k][l[v]]+= 1
                

    info = '\nINFORMATION OF WEISFEIR-LEHMAN REPRESENTATION:\n' + \
                f'  Yo have a total of {len(G)} graphs, each graph\n' + \
                f'  has an Euclidian representation of legth {len(labels_count[0])}.'
    #print(info)
    return labels_count

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

    return (graph, labels) if not grak else grakel.Graph(graph, labels)


def matrix_dist(graphs, n_iter):
    n = len(graphs)
    distance_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            if i == j:
                label_counts = WeisferLheman([graphs[i]], n_iter=n_iter)
                r1,r2 = np.array(list(label_counts[0].values())), np.array(list(label_counts[0].values()))
            else:
                label_counts = WeisferLheman([graphs[i], graphs[j]], n_iter=n_iter)
                r1, r2 = np.array(list(label_counts[0].values())), np.array(list(label_counts[1].values()))
            distance_matrix[i,j], distance_matrix[j,i] = np.dot(r1,r2), np.dot(r1,r2)
            # Print formato carga
        char = f"\033[7m \033[0m"*int(100*i/(n)) + " "*(100-int(100*i/(n)))
        sys.stdout.write(f"\r|{char}|")  # Escribe en la misma línea
        sys.stdout.flush()  # Forzar la actualización de la línea

    f = " "*102 
    sys.stdout.write(f"\r{f}")
    sys.stdout.write(f"\rDistance matrix computed!\n")
    sys.stdout.flush()

    return distance_matrix