r"""
This script reads a graph database and creates a new one
as vectors with the Weisfeirel-Lehman.
"""

import pandas as pd
import numpy as np
import os, sys
import WeisfeirelLehman_m as wl

organism_csv_Prok = pd.read_csv(f"..\\Prok\\data\\Results2.csv", 
                               usecols=["organism","Categories"], 
                               na_values=["nan", "NaN", "NA", ""])
organism_csv_Euk = pd.read_csv(f"..\\Euk\\data\\Results2.csv", 
                               usecols=["organism","Categories"], 
                               na_values=["nan", "NaN", "NA", ""])

        # Eliminam els valor NA
organism_csv_Prok = organism_csv_Prok.dropna()
organism_csv_Euk = organism_csv_Euk.dropna()

n_P =  organism_csv_Prok.shape[0]  # n és el nombre de grafs total
n_E = organism_csv_Euk.shape[0]
n = n_P + n_E
print(f'# Of Organism:{n}\n  Procariotes: {n_P}\n  Eucariotes: {n_E}')
OO = [] # llista auxiliar per tenir dades a un .csv
KK = []  # llista auxiliar per tenir dades a un .csv

data_WL = pd.DataFrame(
    {'G_representation': [], 'Kingdom': []}
)

# Recorrem tota la base de dades i la guardam en una llista amb el següent format
r"""
        FORMAT LLISTA

Cada graf s'ha de guardar en una tupla (G, L) on L és un diccionari amb els atributs 
dels nodes i G el graf amb el format sparse (llista d'arestes amb arestes com a tuples). 
Tots aquests grafs els guardarem en la llista graphs. 
"""

graphs = []
kingdoms = []
X = []
for i in range(n): 
            
    # Prenem un organisme i la classe a la que pertany
    if i < n_P: [organism, kingdom ] = organism_csv_Prok.iloc[i]
    elif i >= n_P: 
        [organism, kingdom ] = organism_csv_Euk.iloc[i-n_P]
    
    # Miram que no sigui una pacteria per no desequilibrar les dades
    if kingdom == 'Bacteria': pass
    else: 
        OO.append(organism)
        KK.append(kingdom)

        arx = f'\\{organism}\\{organism}_mDAG_biggerDAG_adj.csv'
        # Definim la ruta de l'arxiu del organisme
        # Definim la ruta de la carpeta d'on agafarem les dades
        if i < n_P: ruta_org = '..\\Prok\\data\\Individuals'
        else: ruta_org = '..\\Euk\\data\\Individuals'
        ro = ruta_org + arx

        # Lletgim el csv del organisme:
        O_ADJ_csv = pd.read_csv(ro, sep=";", quotechar='"',
                    usecols=["source", "destination"])
        
        # Deinim un diccionari a mode de funció que assigna un nombre a cada node
        nodes = list(set(O_ADJ_csv['source'].to_list()).union(O_ADJ_csv['destination'].to_list()))
        map = {nodo: j for j, nodo in enumerate(nodes)}
        label_kingdom = {'Animals': 0, 'Archaea': 1, 'Plants': 2, 'Fungi': 3, 'Protists': 4}
        
        
        # Escrivim la matriu d'adjacencia en format sparse
        S, D = [], []
        L = {i: 'a' for i in range(len(nodes))}        
        for idx, row in O_ADJ_csv.iterrows():
            o, d = map[O_ADJ_csv.loc[idx,'source']],map[O_ADJ_csv.loc[idx,'destination']] 
            S.append(o)
            D.append(d)
        

        graphs.append(wl.graph_labeled(len(nodes), S, D))

        kingdoms.append(label_kingdom[kingdom])
        
        # Print formato carga
        char = f"\033[7m \033[0m"*int(100*i/n) + " "*(100-int(100*i/n))
        total = '|' + " "*100 + ']'
        sys.stdout.write(f"\r|{char}|")  # Escribe en la misma línea
        sys.stdout.flush()  # Forzar la actualización de la línea



# Guardam en un .csv els organismes i el seu regne    
organism_kingdom = pd.DataFrame({'ORGANISM': OO, 'KINGDOM': KK})
organism_kingdom.to_csv('Weisfeirel-Lehman\\data\\kingdoms.csv',index=False)

n_it = 1
X1 = wl.WeisferLheman(graphs,n_it)
G_reprs = []
for i in range(len(graphs)):
    G_reprs.append(str(list(X1[i].values())))


data_base = pd.DataFrame({
    'Graphs': G_reprs, 'Kingdom': kingdoms
})
data_base.to_csv(f'Weisfeirel-Lehman\\data\\data_base_n{n_it}.csv',index=False)

# Missatge de finalització
f = " "*102 
sys.stdout.write(f"\r{f}")
sys.stdout.write(f"\rData processed!")
sys.stdout.flush()
