import pandas as pd
import os
import sys

"""
Aquest codi és per lletgir les dades. Hi ha dues funcions principals; la primer
prendre les dades que necessitam i guardarles a data_processed; la segona llegeix
aquestes dades i les guarda en tres conjunts: training, validation i test.
"""
def modify_results():
    ruta_actual = os.path.abspath(__file__)
    ruta_superior = os.path.dirname(os.path.dirname(os.path.dirname(ruta_actual)))
    for X in ['Prok', 'Euk']:
        rutita = ruta_superior+f"\\{X}\\data\\Results.csv"
        
        Results_csv = pd.read_csv(rutita, usecols=["organism","Categories"],  na_values=["nan", "NaN", "NA", ""])

        Results_csv = Results_csv.dropna()

        for i in range(len(Results_csv['Categories'])):
            s = Results_csv.iloc[i,1]
        
            if not isinstance(s,str): print('ERROR', s)
            if 'Animals' in s: Results_csv.iloc[i,1] = 'Animals'
            elif 'Protists' in s: Results_csv.iloc[i,1] = 'Protists'
            elif 'Bacteria' in s: Results_csv.iloc[i,1] = 'Bacteria'
            elif 'Fungi' in s: Results_csv.iloc[i,1] = 'Fungi'
            elif 'Archaea' in s: Results_csv.iloc[i,1] = 'Archaea'
            elif 'Plants' in s: Results_csv.iloc[i,1] = 'Plants'

        Results_csv.to_csv(ruta_superior+f"\\{X}\\data\\Results2.csv", index=False)

def process_raw2(mdag = False, largest_component = False):

    r"""
    This functions creates 4 documents .txt that stores the data of the MDAG database.
    Args:
        :param: mdag -> if False it reads reaction graphs, if True it reads the mdag
    """
    # Cream carpetes per guardar els nostres arxius.
    ruta_actual = os.path.abspath(__file__)
    ruta_superior = os.path.dirname(os.path.dirname(os.path.dirname(ruta_actual)))
    
    if mdag: 
        name = 'MDAG'
        if largest_component: name += '_LC'
    else: name = 'RGRAPH'

    destination = os.path.dirname(ruta_actual)
    os.makedirs(destination + f'\\{name}', exist_ok= True)
    

    # Definim es rutes on guardarem les dades
    carpeta_destino = destination + f'\\{name}'
    doc_path_A = carpeta_destino + f'\\{name}_A.txt'
    doc_path_graph_indicator = carpeta_destino + f'\\{name}_graph_indicator.txt'
    doc_path_graph_labels = carpeta_destino + f'\\{name}_graph_labels.txt'
    doc_path_node_labels = carpeta_destino + f'\\{name}_node_labels.txt'

    # Comprovam l'existència dels arxius i els eliminam per evitar problemes:
    if os.path.exists(doc_path_A): os.remove(doc_path_A)
    if os.path.exists(doc_path_graph_indicator): os.remove(doc_path_graph_indicator)
    if os.path.exists(doc_path_graph_labels): os.remove(doc_path_graph_labels)
    if os.path.exists(doc_path_node_labels): os.remove(doc_path_node_labels)

    # Obrim els arxius per escriure
    doc_A = open(doc_path_A, 'w')
    doc_graph_indicator = open(doc_path_graph_indicator, 'w')
    doc_graph_labels = open(doc_path_graph_labels, 'w')
    doc_node_labels = open(doc_path_node_labels, 'w')

    kingdoms = {'Animals': 1, 'Bacteria': None,'Archaea': 0,'Fungi': 0,'Plants': 0,'Protists': 0}
    
    # Definim contadors dels grafs, nodes i arestes que faran falta per escriure en els arxius
    number_graphs = 0
    number_nodes = 0
    number_edges = 0

    for X in ['Prok']:
        organism_csv_Prok = pd.read_csv(ruta_superior+f"\\Prok\\data\\Results2.csv", 
                               usecols=["organism","Categories"], 
                               na_values=["nan", "NaN", "NA", ""])
        organism_csv_Euk = pd.read_csv(ruta_superior+f"\\Euk\\data\\Results2.csv", 
                               usecols=["organism","Categories"], 
                               na_values=["nan", "NaN", "NA", ""])

        # Eliminam els valor NA
        organism_csv_Prok = organism_csv_Prok.dropna()
        organism_csv_Euk = organism_csv_Euk.dropna()

        n_P =  organism_csv_Prok.shape[0]  # n és el nombre de grafs total
        n_E = organism_csv_Euk.shape[0]
        n = n_P + n_E
        for i in range(n): 
            
            # Prenem un organisme i la classe a la que pertany
            if i < n_P: [organism, kingdom ] = organism_csv_Prok.iloc[i]
            elif i >= n_P: [organism, kingdom ] = organism_csv_Euk.iloc[n-i]
            
            if kingdom == 'Bacteria': pass

            else: 
                # Lletgim el reaction graph
                if mdag: 
                    arx = f'\\{organism}\\{organism}_mDAG_adj.csv'
                    if largest_component: arx = f'\\{organism}\\{organism}_mDAG_biggerDAG.csv'
                else: arx = f'\\{organism}\\{organism}_R_adj.csv'

                # Definim la ruta de l'arxiu del organisme
                # Definim la ruta de la carpeta d'on agafarem les dades
                if i < n_P: ruta_org = ruta_superior + '\\Prok\\data\\Individuals'
                else: ruta_org = ruta_superior + '\\Euk\\data\\Individuals'
                ro = ruta_org + arx

                # Print formato carga
                char = f"\033[7m \033[0m"*int(100*i/n) + " "*(100-int(100*i/n))
                total = '[' + " "*100 + ']'
                sys.stdout.write(f"\r[{char}]")  # Escribe en la misma línea
                sys.stdout.flush()  # Forzar la actualización de la línea

                # Escrivim en els nostres arxius

                # Lletgim el csv del organisme:
                O_ADJ_csv = pd.read_csv(ro, sep=";", quotechar='"',
                                    usecols=["source", "destination"])
                number_graphs += 1 # Ens situam en un nou graf
                label = kingdoms[kingdom]
                doc_graph_labels.write(str(label) + '\n')
                
                # Deinim un diccionari a mode de funció que assigna un nombre a cada node
                nodes = list(set(O_ADJ_csv['source'].to_list()).union(O_ADJ_csv['destination'].to_list()))
                map = {nodo: number_nodes + j + 1 for j, nodo in enumerate(nodes)}
                inverse_map = {number_nodes + j + 1: nodo for j, nodo in enumerate(nodes)}
            
                # Escrivim la matriu d'adjacencia
                for idx, row in O_ADJ_csv.iterrows():
                    doc_A.write(f'{map[row['source']]}, {map[row['destination']]}\n')
    
                # Escrivim a quin graf pertany el node
                for node in nodes:
                    doc_graph_indicator.write(f'{number_graphs}\n')
                    doc_node_labels.write(f'1\n')
                    number_nodes += 1

        f = " "*102 
        sys.stdout.write(f"\r{f}")
        sys.stdout.write(f"\rData processed!")
        sys.stdout.flush()

    # Print data cargado
   
    doc_A.close()
    doc_graph_indicator.close()
    doc_graph_labels.close()
    doc_node_labels.close()

modify_results()
#process_raw2(mdag=True)