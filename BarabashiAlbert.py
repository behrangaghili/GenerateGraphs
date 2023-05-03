import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# real network metrics 
"""_summary_
Number of nodes: 4039
Number of edges: 88234
Average degree: 43.69
Density: 0.0108
Diameter: 8
Average clustering coefficient: 0.6055
Transitivity: 0.5192
Average shortest path length: 3.69
Assortativity (Degree Correlation): 0.0636
max Degree: 1045
max Betweenness: 0.4805180785560152
max Closeness: 0.45969945355191255
Network Centralization based on Degree Centrality: 6.143991253825224e-05
Network Centralization based on Betweenness Centrality: 0.00011889273100267206
Network Centralization based on Closeness Centrality: 4.5473653981117466e-05
Network Centralization based on PageRank Centrality: 1.8253220774909598e-06
"""

# Barabási-Albert model parameters
n = 4039
m0 = 22
m = 22

# Create the initial graph with m0 nodes
G = nx.complete_graph(m0)

# Save the graph as an adjacency list in a text file
nx.write_adjlist(G, "watts_strogatz_graph_adjlist.txt")

# Add the remaining nodes with m edges to existing nodes
for i in range(m0, n):
    # Calculate the probabilities for each existing node
    degrees = [G.degree(node) for node in G.nodes()]
    total_degree = sum(degrees)
    probabilities = [degree / total_degree for degree in degrees]

    # Select m nodes to attach the new node to
    selected_nodes = np.random.choice(G.nodes(), size=m, replace=False, p=probabilities)

    # Attach the new node to the selected nodes
    for node in selected_nodes:
        G.add_edge(i, node)
      
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
print("done")
# # Visualize the graph
# pos = nx.spring_layout(g)
# nx.draw(g, pos, with_labels=true, node_color='blue', edge_color='gray', node_size=100, font_size=8)
# plt.show()
