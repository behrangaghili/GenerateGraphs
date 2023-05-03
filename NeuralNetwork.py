import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Define the graph generator model using PyTorch
class GraphGenerator(nn.Module):
    def __init__(self, num_nodes, num_features):
        super(GraphGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.lin1 = nn.Linear(num_nodes*num_nodes, 64)
        self.lin2 = nn.Linear(64, num_nodes*num_nodes)
    
    def forward(self, x):
        x = x.view(1, -1)
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        x = x.view(-1, self.num_nodes, self.num_nodes)
        return x

# Define a function to calculate various metrics of a graph
def calculate_metrics(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    metrics = {}
    metrics['Density'] = nx.density(G)
    metrics['Clustering Coefficient'] = nx.average_clustering(G)
   # metrics['Diameter'] = nx.diameter(G)
    metrics['Average Degree'] = np.mean(np.sum(adj_matrix, axis=0))
    metrics['Transitivity'] = nx.transitivity(G)
    metrics['Degree Correlation'] = nx.degree_assortativity_coefficient(G)
    print(metrics)
    return metrics

# Define the training loop
def train(model, optimizer, criterion, adj_matrix, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        adj_matrix_pred = model(adj_matrix)
        adj_matrix_pred = adj_matrix_pred.view(1, -1)
        adj_matrix_reshaped = adj_matrix.view(1, -1)
        loss = criterion(adj_matrix_pred, adj_matrix_reshaped)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# Load the input graph from a text file
filename = 'graph.txt'
df = pd.read_csv(filename, sep='\t', header=None, names=['FromNodeId', 'ToNodeId'])
df['FromNodeId'] = pd.factorize(df['FromNodeId'])[0]
df['ToNodeId'] = pd.factorize(df['ToNodeId'])[0]
num_nodes = max(df[['FromNodeId', 'ToNodeId']].max()) + 1
adj_matrix_input = np.zeros((num_nodes, num_nodes))
for i, row in df.iterrows():
    adj_matrix_input[row['FromNodeId']][row['ToNodeId']] = 1

# Define the graph generator model and optimizer
num_features = 64
model = GraphGenerator(num_nodes, num_features)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model to generate similar graphs
num_epochs = 100
losses = train(model, optimizer, criterion, torch.FloatTensor(adj_matrix_input).view(1, -1), num_epochs)


# Generate a new graph using the trained model
adj_matrix_output = model(torch.zeros((1, num_nodes*num_nodes))).detach().numpy()[0]

# Calculate the metrics of the input and output graphs
metrics_input = calculate_metrics(adj_matrix_input)
metrics_output = calculate_metrics(adj_matrix_output)

# Print the metrics of the input and output graphs
print('Input Graph Metrics:')
for key, value in metrics_input.items():
    print(key, ':', value)

print('\nGenerated Graph Metrics:')
for key, value in metrics_output.items():
    print(key, ':', value)

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# Plot the input and output graphs
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(adj_matrix_input, cmap='gray_r')
axs[0].set_title('Input Graph')
axs[1].imshow(adj_matrix_output, cmap='gray_r')
axs[1].set_title('Generated Graph')
plt.show()

#Calculate the similarity between the input and output graphs

similarity = 0
for key in metrics_input.keys():
    similarity += abs(metrics_input[key] - metrics_output[key])
    similarity /= len(metrics_input)

print('\nThe similarity between the input and output graphs is:', similarity)