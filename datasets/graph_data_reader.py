import networkx as nx
import numpy as np
import os
import math
'''
Carregam el paquet os.path.join() perquè treballarem
amb moltes carpetes i diferents rutes per obrir arxius. 
Així evitam problemes de format en diferents ordinadors.
'''
from os.path import join as pjoin
import copy

import torch


"""
References: https://github.com/bknyaz/graph_nn
"""

class DataReader():

    def __init__(self,
                 data_dir, 
                 rnd_state=None,
                 folds=10):
        
        r"""
        Cream un objecte DataReader que consta de:
            > data_dir: Carpeta on es troben els arxius
            > rnd_state: Llavor per tots els processos aleatoris
            > data: diccionari amb la informació de la base de dades
            > folds
        """

        self.data_dir = data_dir

        # Definim un rnd_state per recuperar resultats
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state

        """
        Lletgim els arxius que estan en el dataset DS:
            > DS_A.txt
            > DS_graph_indicator.txt
            > DS_graph_labels.txt
            > DS_node_labels.txt
        """
        files = os.listdir(self.data_dir)
        
        print('data path:', self.data_dir)
        
        data = {}
        
        """
        Cridam la funció read_graph_nodes_relations amb  argument DS_graph_indicator.txt
        Aquesta guarda en nodes un diccionari on per cada node s'indica el graf que pertany i en graphs
        un diccionari on a cada graph li assigna el seu conjunt de nodes en forma de llista
        """
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0]) 

        # Gurdam en el diccionari una llista amb les matrius d'adjacència
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)        
                
        print('complete to build \033[1mADJACENCY MATRIX\033[0m list')
        

        # Make node count list
        data['node_count_list'] = self.get_node_count_list(data['adj_list'])
        
        print('complete to build node count list')

        # Make edge matrix list
        data['edge_matrix_list'], data['max_edge_matrix'] = self.get_edge_matrix_list(data['adj_list'])
        
        print('complete to build edge matrix list')

        # Make node count list
        data['edge_matrix_count_list'] = self.get_edge_matrix_count_list(data['edge_matrix_list'])
        
        print('complete to build edge matrix count list')
        
        # Make degree_features and max neighbor list
        degree_features = self.get_node_features_degree(data['adj_list'])
        data['max_neighbor_list'] = self.get_max_neighbor(degree_features)
        
        print('complete to build max neighbor list')
       
        # Read features or make features
        if len(list(filter(lambda f: f.find('node_labels') >= 0, files))) != 0:
            print('node label: node label in dataset')
            data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0], 
                                                     nodes, graphs, fn=lambda s: int(s.strip()))
        else:
            print('node label: degree of nodes')
            data['features'] = degree_features
            
        print('complete to build node features list')
        
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))
                                                       
        print('complete to build targets list')
        
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj) # Number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj) # Total sum of edges
            n_edges.append( int(n / 2) ) # Undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))
                        
        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1) # Number of possible values
        
        features_onehot = []
        for i, x in enumerate(features):
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            features_onehot.append(feature_onehot)
            
        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets'] # Graph class labels
        labels -= np.min(labels) # To start from 0
        N_nodes_max = np.max(shapes)

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)
        
        print('-'*50)
        print('The number of graphs:', len(data['adj_list']))
        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        for u in np.unique(features_all):
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # Number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        # Create test sets first
        train_ids, test_ids = self.split_ids(np.arange(N_graphs), rnd_state=self.rnd_state, folds=folds)

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
        #################################
        ####### !!!!!!!!!!!!!!! #########
        #################################
        # Aquests els feim servir?

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # Max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes
        self.data = data

    def split_ids(self, ids_all, rnd_state=None, folds=10):
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids


    def parse_txt_file(self, fpath, line_parse_fn=None):
        r"""
        Funció per transformar un arxiu en una llista on cada element
        es una linia amb un format definit  per la funció line_parse_fn
        
        Args:
            fpath -> ubicació del arxiu
            line_parse_fn -> funció que especifica format de la linia
        Return:
            list 
        """

        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):
        r"""
        Aquesta funcio llegeix la matriu d'adjacencia i la retorna en format llista.

        Args:
            fpath -> on es troba l'arxiu + nom
            nodes -> diccionari node -> graf
            graphs -> diccionari graf -> nodes

        Return:
            Llista de matrius d'adjacencia en format np.matrix.
        """

        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
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
        
    def read_graph_nodes_relations(self, fpath):
        r'''
        Funció per lletgir graph_indicator.txt retornant dos diccionaris.
        El primer (nodes) node_id -> graph_id 
        i el segon (graphs) graph_id -> list(nodes_id)
        '''
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
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

    def read_node_features(self, fpath, nodes, graphs, fn):
        r'''
        Funció que llegeix node_attributes.txt i retorna una llista de 
        llistes dels atributs dels nodes. És a dir, els atributs del graf
        i-èssim estan en la posició i-èssima d'aquesta llista (si indexam 
        els grafs desde 0)

        Args:
            fpath -> on es troba l'arxiu DS_node_attributes o features?
            nodes -> diccionari node -> graf
            graphs -> diccionari graf -> nodes
            fn -> funció separació features
        Returns:
            Llista de llistes amb els atributs dels nodes de cada graf. 
        '''
        # Lletgim l'arxiu dels node features
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)

        # Cream un diccionario on per cada key (graph_id) li associam una llista de node_features
        node_features = {}
        
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])

            # Lletgim l'índex del node en el seu graf
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]

            # Afegim en aquella posició el node feature que estava en el arxiu
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

    def get_node_features_degree(self, adj_list):
        r"""
        Lletgeix els graus de cada node

        Args:
            adj_list -> llista amb les matrius d'adjacència
        
        Return:
            Una llista de np.arrays dels graus de cada node 
        """
        node_features_list = []
 
        for adj in adj_list:
            sub_list = []
        
            for feature in nx.from_numpy_array(np.array(adj)).degree():
                sub_list.append(feature[1])
            node_features_list.append(np.array(sub_list))

        return node_features_list
     
    def get_max_neighbor(self, degree_list):
        """
        Args:
            degree_list -> llista de graus dels nodes 
            (get_node_features)
        Return:
            Una llista del grau maxim en cada graf
        """
        max_neighbor_list = []
        
        for degrees in degree_list:
            max_neighbor_list.append(int(max(degrees)))

        return max_neighbor_list

    # Aquesta funció és un poc inutil ja que amb graph_nodes_relations ja ho podem treure
    def get_node_count_list(self, adj_list):
        r"""
        Args:
            adj_list -> llista de matrius d'adjacència
        Return:
            Llista de longituds grafs
        """
        node_count_list = []
        
        for adj in adj_list:
            node_count_list.append(len(adj))
                        
        return node_count_list

    def get_edge_matrix_list(self, adj_list):
        r"""
        Args:
            adj_list -> llista de les matrius d'adjacència
        Returns:
            edge_matrix_list -> llista de llistes de les arestes
            max_edge_matrix -> maxim nombre d'arestes
        """
        edge_matrix_list = []
        max_edge_matrix = 0
        
        for adj in adj_list:
            edge_matrix = []
            for i in range(len(adj)):
                for j in range(len(adj[0])):
                    if adj[i][j] == 1:
                        edge_matrix.append((i,j))
            if len(edge_matrix) > max_edge_matrix:
                max_edge_matrix = len(edge_matrix)
            edge_matrix_list.append(np.array(edge_matrix))
                        
        return edge_matrix_list, max_edge_matrix

    def get_edge_matrix_count_list(self, edge_matrix_list):
        r"""
        Args:
            edge_matrix_list -> llista de llistes de les arestes
        Return:
            llista de nombre d'arestes
        """
        edge_matrix_count_list = []
        
        for edge_matrix in edge_matrix_list:
            edge_matrix_count_list.append(len(edge_matrix))
                        
        return edge_matrix_count_list


class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 fold_id,
                 datareader,
                 split):
        
        
        self.set_fold(datareader.data, split, fold_id)

    def set_fold(self, data, split, fold_id):
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.max_edge_matrix = data['max_edge_matrix']
        self.n_classes = data['n_classes']
        self.features_dim = data['features_dim']
        self.idx = data['splits'][fold_id][split]
        
        # Use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
        self.max_neighbor_list = copy.deepcopy([data['max_neighbor_list'][i] for i in self.idx])
        self.edge_matrix_list = copy.deepcopy([data['edge_matrix_list'][i] for i in self.idx])
        self.node_count_list = copy.deepcopy([data['node_count_list'][i] for i in self.idx])
        self.edge_matrix_count_list = copy.deepcopy([data['edge_matrix_count_list'][i] for i in self.idx])
        self.imag_lapl = None
        self.imag2 = 'hola!'

        

        
        print('%s: %d/%d' % (split.upper(), len(self.labels), len(data['targets'])))
        
        # Sample indices for this epoch
      
        self.indices = np.arange(len(self.idx))
        
    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0, mode='edge_matrix'):
        sz = mtx.shape
        #assert len(sz) == 2, ('only 2d arrays are supported', sz)
        
        if len(sz) == 2:
            if desired_dim2 is not None:
                  mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)
            else:
                  mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        elif len(sz) == 3:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0), (0, 0)), 'constant', constant_values=value)
        
        return mtx
    
    def nested_list_to_torch(self, data):
        #if isinstance(data, dict):
            #keys = list(data.keys())           
        for i in range(len(data)):
            #if isinstance(data, dict):
                #i = keys[i]
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            #elif isinstance(data[i], list):
                #data[i] = list_to_torch(data[i])
        return data
        
    r"""
    Les funcions __len__ i __getitem__ son necessàries per 
    poder definir posteriorment un loader.
    """
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = self.indices[index]
        N_nodes_max = self.N_nodes_max
        N_nodes = self.adj_list[index].shape[0]
        graph_support = np.zeros(self.N_nodes_max)
        graph_support[:N_nodes] = 1
        breakpoint()
        return self.nested_list_to_torch([self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # Node_features
                                    self.pad(self.adj_list[index], self.N_nodes_max, self.N_nodes_max),  # Adjacency matrix
                                    self.pad(self.imag_lapl[index], self.N_nodes_max, self.N_nodes_max), # Imag part
                                    graph_support,  # Mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
                                    N_nodes,
                                    int(self.labels[index]),
                                    int(self.max_neighbor_list[index]),
                                    self.pad(self.edge_matrix_list[index], self.max_edge_matrix),
                                    int(self.node_count_list[index]),
                                    int(self.edge_matrix_count_list[index])])     

    def ad2MagLapl(self, q,  normalized = True):
        N = len(self.adj_list)
        self.imag_lapl = []
        for i in range(N):
            real, imag = GraphData.ad2MagL(self.adj_list[i], q,  normalized)
            self.adj_list[i] = real
            self.imag_lapl.append(imag)

    @staticmethod
    def ad2MagL(Adj, q, normalized):

        As = 0.5*(Adj + Adj.T)
        D = [np.sum(As[i]) for i in range(len(As))]
        D_norm = np.diag([np.power(D[i], -0.5) if D[i] != 0 else 0 for i in range(len(D))])
        T = 2*math.pi*q*(Adj - Adj.T) 
        T = np.cos(T) + np.sin(T)*1.0j
        I = np.eye(len(Adj)) 

        if normalized:
            L_n = I - (np.matmul(np.matmul(D_norm, As),D_norm))*T
            L_n_real, L_n_imag = L_n.real, L_n.imag 
            return L_n_real, L_n_imag
        else:
            L =  np.diag(D) -  As*T 
            return L
    
    def __repr__(self):
        return super().__repr__()
    