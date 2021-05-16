import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
from collections import Counter
import seaborn; seaborn.set()

# %matplotlib inline
data  = [[1,14], [2,17], [2,18], [2,19],[2,20], [2,21], [2,28],[3,16],[3,17], [3,18], [5,1],
         [5,13], [5,15], [5,16], [5,28],[6,15], [7,3],  [7,20],[7,21],[7,22], [8,22], [9,0],
         [9,16], [9,19], [9,22], [9,23],[9,24], [9,25], [9,26],[9,28],[10,13],[10,22],[10,24],
         [10,25],[10,26],[10,27],[11,0],[11,23],[11,27],[12,0],[12,1],[12,4], [12,14],[12,28]]


G = nx.Graph()

G.add_edges_from(data)

fig, ax = plt.subplots(figsize=(16, 9))

nx.draw(G, with_labels=True)
plt.show()

G.number_of_nodes()

G.number_of_edges()


def local_clustering(G, nodes=None):
    td_iter = _triangles_and_degree_iter(G, nodes)
    clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for v, d, t, _ in td_iter}
    if nodes in G:
        return clusterc[nodes]
    return clusterc

def global_clustering(localc):
    c1 = 0
    c2 = 0

    for name, dict_ in localc.items():
        c1 += dict_
        c2 +=1

    return c1/c2


lc = local_clustering(G)

for name, dict_ in lc.items():
    if dict_ != 0:
        print("Node: ", name, "\n C_i", dict_,"\n")

gc = global_clustering(lc)

print("Global Cluster Coeficient:", gc)

def average_shortest_distance(G):
    n = len(G)
    s = sum(l for u in G for l in nx.single_source_shortest_path_length(G, u).values())
    return s / (n *(n-1))


print(average_shortest_distance(G))

def centrality(G):
    if len(G) <= 1:
        return {n: 1 for n in G}

    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for n, d in G.degree()}
    return centrality

cnt = centrality(G)
cnt

r = nx.degree_assortativity_coefficient(G)

print(r)
