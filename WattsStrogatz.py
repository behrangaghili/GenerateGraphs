import networkx as nx
import matplotlib.pyplot as plt

# Parameters for the Watts-Strogatz graph
n = 2777   # Number of nodes
k = 13     # Each node is connected to k nearest neighbors in a ring topology
p = 0.1  # Probability of rewiring each edge

# Generate the graph
G = nx.watts_strogatz_graph(n, k, p)

# Save the graph as an adjacency list in a text file
nx.write_adjlist(G, "watts_strogatz_graph_adjlist.txt")


# Compute the basic network properties
num_nodes = len(G.nodes())
num_edges = len(G.edges())
avg_degree = sum([val for (node, val) in G.degree()]) / float(num_nodes)
density = nx.density(G)

# As the graph is not connected, I am reporting here the largest computed diameter among the connected components.
# The commented code below gets an exception "Found infinite path length because the graph is not connected"
# Therefore, I used the connected component with the maximum diameter.
#diameter = nx.diameter(G.to_undirected())
diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(G)])
avg_clustering = nx.average_clustering(G)
transitivity = nx.transitivity(G)
avg_shortest_path_length = nx.average_shortest_path_length(G)

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Average degree: {avg_degree:.2f}")
print(f"Density: {density:.4f}")
print(f"Diameter: {diameter}")
print(f"Average clustering coefficient: {avg_clustering:.4f}")
print(f"Transitivity: {transitivity:.4f}")
print(f"Average shortest path length: {avg_shortest_path_length:.2f}")

# Compute assortativity
assortativity = nx.degree_assortativity_coefficient(G)
print(f"Assortativity (Degree Correlation): {assortativity:.4f}")

# compute centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
pagerank_centrality = nx.pagerank(G)
# Compute network centralization
max_degree = max([val for (node, val) in G.degree()])
max_betweenness = max(betweenness_centrality.values())
max_closeness = max(closeness_centrality.values())
max_pagerank = max(pagerank_centrality.values())
degree_centralization = sum([(max_degree - val) for (node, val) in G.degree()]) / float((num_nodes - 1) * (num_nodes - 2))
betweenness_centralization = sum([(max_betweenness - val) for (node, val) in betweenness_centrality.items()]) / float((num_nodes - 1) * (num_nodes - 2))
closeness_centralization = sum([(max_closeness - val) for (node, val) in closeness_centrality.items()]) / float((num_nodes - 1) * (num_nodes - 2))
pagerank_centralization = sum([(max_pagerank - val) for (node, val) in pagerank_centrality.items()]) / float((num_nodes - 1) * (num_nodes - 2))

# Print network centralization
print(f"max degree =", max_degree)
print("max betweenness=", max_betweenness)
print("max closeness=",max_closeness )
print("max pagerank = ",max_pagerank )
print("degree centralization = ",degree_centralization )
print("betweenness centralization = ",betweenness_centralization )
print("closeness centralization = ",closeness_centralization )
print("pagerank centralization = ",pagerank_centralization )

# Plot degree distribution
degrees = [val for (node, val) in G.degree()]
plt.hist(degrees, bins=50)
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()

# # Visualize the graph
# pos = nx.spring_layout(g)
# nx.draw(g, pos, with_labels=true, node_color='blue', edge_color='gray', node_size=100, font_size=8)
# plt.show()
