import numpy as np
import os 
from os.path import join as pjoin
import sys

r"""
This document computes the node features (d, d_in, d_out) of the nodes
"""
ruta_actual = os.path.dirname(os.path.abspath(__file__))
data_dir = ruta_actual + '\\RGRAPH'
doc_node_features = open(data_dir + '\\RGRAPH_node_labels.txt', 'w')

# We create the dictionary node->graphs; graphs-> nodes

def parse_txt_file(fpath, line_parse_fn=None):
        r"""
        Funció per transformar un arxiu en una llista on cada element
        es una linia amb un format definit  per la funció line_parse_fn
        
        Args:
            fpath -> ubicació del arxiu
            line_parse_fn -> funció que especifica format de la linia
        Return:
            list 
        """

        with open(pjoin(data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data
    
def read_graph_adj(fpath, nodes, graphs):
        r"""
        Aquesta funcio llegeix la matriu d'adjacencia i la retorna en format llista.

        Args:
            fpath -> on es troba l'arxiu + nom
            nodes -> diccionari node -> graf
            graphs -> diccionari graf -> nodes

        Return:
            Llista de matrius d'adjacencia en format np.matrix.
        """

        edges = parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]

            # Verificam que els dos nodes són del mateix graf
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                # Hem de definir un nou graf
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))

            # Calculam el índex on estan ubicats el nostre node            
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]

            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1

        # Retornam les matrius d'adjacència en format llista 
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        
        return adj_list
        
def read_graph_nodes_relations(fpath):
        r'''
        Funció per lletgir graph_indicator.txt retornant dos diccionaris.
        El primer (nodes) node_id -> graph_id 
        i el segon (graphs) graph_id -> list(nodes_id)
        '''
        graph_ids = parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

nodes, graphs = read_graph_nodes_relations('RGRAPH_graph_indicator.txt')
list_adj = read_graph_adj('RGRAPH_A.txt', nodes, graphs)

for A in list_adj:
    N = len(A) # Number of nodes

    # It should be in order...
    
    for i in range(N):
        # features for node i on the graph A
        d = 0
        d_in = 0
        d_out = 0
        for j in range(N):
            if A[i][j] == 1:  
                d += 1
                d_in += i < j
                d_out += i > j
        s = f'{d},{d_in},{d_out}\n'
        doc_node_features.write(s)
doc_node_features.close()    