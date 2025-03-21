# README




### Modificacions

Hem hagut de crear un enviorment de `Python 3.12` amb els següents paquets:

-   `torch`: Un paquet de python que blalblabla
-   `torch_geometric`: Un paquet per treballar amb grafs i GNNs
-   `torch_summary`: ?¿¿?¿?
-   `networkx` 
-   `numpy`

A més hem hagut de modificar algunes funcions de `networkx`, doncs ja eren antigues. En concret la funció `X` per `Y`

## Funcionament amb `ArgParse` 

`argparse` és una llibreria que permet modificar arguments desde la "linia de comandos". En el nostre cas treballarem desde'l **Anaconda PowerShell Prompt** per evitar problemes amb paquets; d'aqui en endavant sempre treballarem en aquesta terminal amb l'enviorment activat. Així, desde la terminal podrem introduir els models que volem entrenar, el nombre de folds,  el nombre de capes i d'èpoques entre d'altres. Així com si volem que els resultats es guaardin o no.

Per exemple per saber com funciona el train.py podem escriure en la terminal `python train.py --help`, i així obtindrem quines variables podem modificar, quins valors podem escriure i com ho hem de fer.

## Data Reader

En aquesta secció explicarem com es llegeixen les dades per preparar-les per entrenar un model. Per el correcte funcionament amb tots els models i mantenir l'estructura d'altres codis, eprarem el format de: [bknyaz](https://github.com/bknyaz/graph_nn). En aquest format, útil per dades reals al tractar-se de grafs dispersos, només escriurem les arestes, i en altres arxius altres atributs que necessitarem. Doncs un conjunt de dades, vendra definit per 5 arxius, que contindran tota la informació. Per les dades "DS" tenim els següents arxius:

-   **DS_A.txt**: Aquest arxiu guarda totes les arestes de tots els grafs. Les arestes venen especificades per un node inicial i un final separats per una coma. Els nodes és començan a contar desde el graf 1 fins el darrer. Llavors sabrem si una aresta és de un graf o un altre en funció del nombre dels nodes, tot coneguent les longituds dels grafs i els seus indexos. Per exemple si sabem que el graf 1 té 40 nodes i tenim una aresta "41, 55" sabem que pertany al graf 2.

-   **DS_graph_indicator.txt**: Començant del 1 fins al nombre de grafs, escriu tantes vegades aquest index com nodes tengui el graf. 

-   **DS_graph_labels.txt**: En la linia i-èssima s'escriu la classe a la que pertany el graf i-èssim per la classificació.

-   **DS_node_attributes.txt**: En la linia i-èssima s'escriu l'atriubut del node i-èssim (un nombre real)

-   **DS_node_labels.txt**: En la linia i-èssima s'escriu la classe del node i-èssim.

És important que totes les dades tenguin aquest format per poder comparar els models amb diferents àmbits. L'únic que pot variar és el nom "DS", doncs per exemple si trebajam amb el conjunt de dades "PROTEINS" hem de tenir els següents arxius:
**PROTEINS_A.txt**, **PROTEINS_graph_indicator.txt**, **PROTEINS_graph_labels.txt**, **PROTEINS_node_attributes.txt** i **PROTEINS_node_labels.txt**; tots guardats en una carpeta **datasets/PROTEINS**.

### Funcionament *graph_data_reader.py*

Aquest programa defineix una classe `DataReader`, aquesta clase permet crear un objecte que utilitzarem per lletgir  unes dades en concret. Per diferents dades crearem diferents *DataReaders*. Explicarem les funcions i atributs principals que necessitarem.

Al guardar totes les 'matrius d'adjecència' en un mateix document .txt hem de definir bé com les lletgirem per evitar solapaments amb dades d'altres grafs. Per això definim la funció `read_graph_adj` de paràmetres `fpath` (on estan ubicades les dades), `nodes` (un diccioneari que indica a quin graf està aquell node) i `graph` (un diccionari de claus *graph_id* relacionats amb els nodes?). Aquesta funció retorna una llista amb totes les matrius d'adjacència. 

Aquesta funció s'ajuda de `parse_txt_file` la qual com indica el seu nom analitza un arxiu .txt i el retorna en un format manejable per python. Rebrà com a pràmetres `fpath`(file path) i `line_parse_fn` (una funció que indica com s'ha d'analitzar una linia). Aquesta funció retorna una llista on cada element és la imatge per `line_parse_fn` de cada linia. 

Llavors, tornant a la funció anterior, llegeix totes les arestes i amb `line_parse_fn = lambda s: s.split(',')`. Per tant crea un diccionari `adj_dict`, on tenim de claus els *graph_id* relacionat amb la matriu d'adjacència, que es van completant a mesura que es llegeixen arestes. Finalment transforma aquest diccionari en una llista de matrius. La resta de funcions tenen un funcionament similar, formant el següent conjunt de funcions:

-   `read_graph_adj`
-   `read_graph_nodes_relations`
-   `read_node_features`
-   `get_node_features_degree`
-   `get_max_neighbor`
-   `get_node_count_list`
-   `get_edge_matrix_list`
-   `get_edge_matrix_count_list`
-   `get_neighbor_dic_list`

Amb ajuda d'aquestes funcions podem definir un objecte **DataReader** amb els atributs necesseris per l'entrenament de les dades. 

## Loaders 
## Redaouts i darreres capes

Una complicació que sorgeix quan treballam amb una GNN a nivell de grafs és com feim el readout. Doncs treballam amb un input que consisteix amb un **Graf** i un conjunt d'**atributs dels nodes**. Doncs un pot arribar a entendre com funciona el pas de la informació al llarg de una GNN de qualsevol tipus; però entendre les darreres passes pot ser gaire complicat. Per entendrer-ho bé, començarem analitzan una GCN.

### Funcionament *Graph Convolutional Neural Network*

# MagNet