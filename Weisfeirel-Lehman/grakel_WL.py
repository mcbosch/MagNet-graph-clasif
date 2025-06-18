from grakel import GraphKernel
from grakel.kernels import WeisfeilerLehman,VertexHistogram
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import WeisfeirelLehman_m as wl

import seaborn as sns


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
g_list = []
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

        arx = f'\\{organism}\\{organism}_mDAG_adj.csv'
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
        
        graphs.append((list(zip(S,D)),L))
        kingdoms.append(label_kingdom[kingdom])
        
        # Print formato carga
        char = f"\033[7m \033[0m"*int(100*i/n) + " "*(100-int(100*i/n))
        total = '|' + " "*100 + ']'
        sys.stdout.write(f"\r|{char}|")  # Escribe en la misma línea
        sys.stdout.flush()  # Forzar la actualización de la línea

f = " "*102 
sys.stdout.write(f"\r{f}")
sys.stdout.write(f"\rData processed!\n")
sys.stdout.flush()


G, y = graphs, kingdoms

emparejados_sorted = list(sorted(zip(y, G), key=lambda x: x[0]))

G_ord = [g for _, g in emparejados_sorted]
y_ord = [i for i, _ in emparejados_sorted]


# Usa el kernel VertexHistogram (basado en etiquetas de nodos)
gk = WeisfeilerLehman(n_iter=5,base_graph_kernel=VertexHistogram,normalize=True)
print(gk)
# Calcula la matriz de semejanza
K = gk.fit_transform(G_ord)
print(K)
dK = pd.DataFrame(K, columns=[f'g{i}' for i in range(len(G_ord))])
labels = pd.DataFrame(y.sort(), columns=[f'g{i}' for i in range(len(G_ord))])
dK.to_csv('kernel_matrix.csv')
labels.to_csv('labels.csv')
palette = sns.color_palette("pastel", 5)  # 6 colores diferentes

colores_etiquetas = [palette[i] for i in y_ord]  # Asigna color a cada columna
fig = plt.figure(figsize=(10,10))

ax = sns.heatmap(K,
            cmap="Reds",
            cbar=True,
            xticklabels=False,
            yticklabels=False)



plt.title('Semblances m-DAG')
plt.tight_layout()
plt.savefig('heat_map_kernels.png')
plt.show()
