import networkx as nx

from ridley.ConceptNetAPiAccess import *

G = nx.Graph()
e = GetEntity("dog")
G.add_node(e)
nx.draw(G)
