import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATNet, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads)
        self.linear = nn.Linear(out_channels * heads, out_channels)

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        return x

def construct_graph(patches, num_clusters, k=5):
    device = patches.device
    batch_size, num_patch, nvars, patch_len = patches.shape
    patches_reshaped = patches.view (batch_size * num_patch, nvars * patch_len).cpu ().numpy ()

    # Using k-means for clustering
    kmeans = KMeans (n_clusters=num_clusters, random_state=0).fit (patches_reshaped)
    labels = kmeans.labels_

    # Create a list to hold edges
    edge_index = []

    # Calculate pairwise distances within each cluster and keep k-nearest neighbors
    for cluster_id in range (num_clusters):
        cluster_indices = np.where (labels == cluster_id)[0]
        cluster_patches = patches_reshaped[cluster_indices]

        # Calculate pairwise distances within the cluster
        distances = euclidean_distances (cluster_patches)

        # Find the k-nearest neighbors for each node in the cluster
        for i in range (len (cluster_indices)):
            neighbors = np.argsort (distances[i])[1:k + 1]  # Exclude the node itself
            for neighbor in neighbors:
                edge_index.append ([cluster_indices[i], cluster_indices[neighbor]])

    edge_index = torch.tensor (edge_index, dtype=torch.long).t ().contiguous ().to (device)
    patches_reshaped = torch.tensor (patches_reshaped, dtype=torch.float).to (device)
    return patches_reshaped, edge_index, labels


def contrastive_loss(z_i, z_j, temperature=0.5):
    # Compute cosine similarity
    cos_sim = F.cosine_similarity (z_i.unsqueeze (1), z_j.unsqueeze (0), dim=2)
    sim_matrix = torch.exp (cos_sim / temperature)

    # Positive pairs (diagonal)
    pos_sim = torch.diag (sim_matrix)

    # Compute loss
    loss = -torch.log (pos_sim / sim_matrix.sum (dim=1)).mean ()
    return loss


class GraphContrastiveNetwork (nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, num_clusters=5, k=5):
        super (GraphContrastiveNetwork, self).__init__ ()
        self.num_clusters = num_clusters
        self.k = k
        self.gat_net = GATNet (in_channels, out_channels, heads)

    def forward(self, patches):
        patches_reshaped, edge_index, labels = self.construct_graph (patches)

        # Forward pass through GAT
        patch_representations = self.gat_net (patches_reshaped, edge_index)

        # Reshape back to original shape
        batch_size, num_patch, nvars, patch_len = patches.shape
        patch_representations = patch_representations.view (batch_size, num_patch, nvars, patch_len)

        return patch_representations

    def construct_graph(self, patches):
        return construct_graph (patches, self.num_clusters, self.k)