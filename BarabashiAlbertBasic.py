import networkx as nx
import matplotlib.pyplot as plt

# Parameters for the Barabasi-Albert graph
n = 4039  # Number of nodes
m = 22    # Number of edges to attach from a new node to existing nodes

# Generate the graph
G = nx.barabasi_albert_graph(n, m)

# Save the graph as an adjacency list in a text file
nx.write_adjlist(G, "Barabash_Albert_graph_adjlist.txt")
# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='blue', edge_color='gray', node_size=100, font_size=8)
plt.show()
