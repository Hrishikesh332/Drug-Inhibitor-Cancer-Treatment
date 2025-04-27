import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class GraphDegreeConv(nn.Module):
    def __init__(self, node_size, edge_size, output_size, degree_list, device, batch_normalize=True):
        super(GraphDegreeConv, self).__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.output_size = output_size
        self.batch_normalize = batch_normalize
        self.device = device
        if self.batch_normalize:
            self.normalize = nn.BatchNorm1d(output_size, affine=False)
        self.bias = nn.Parameter(torch.zeros(1, output_size))
        self.linear = nn.Linear(node_size, output_size, bias=False)
        self.degree_list = degree_list
        self.degree_layer_list = nn.ModuleList()
        for degree in degree_list:
            self.degree_layer_list.append(nn.Linear(node_size + edge_size, output_size, bias=False))

    def forward(self, graph, node_repr, edge_repr, neighbor_by_degree):
        degree_activation_list = []
        for d_idx, degree_layer in enumerate(self.degree_layer_list):
            degree = self.degree_list[d_idx]
            node_neighbor_list = neighbor_by_degree[degree]['node']
            edge_neighbor_list = neighbor_by_degree[degree]['edge']
            if degree == 0 and node_neighbor_list:
                zero = torch.zeros(len(node_neighbor_list), self.output_size).to(self.device).float()
                degree_activation_list.append(zero)
            else:
                if node_neighbor_list:
                    # (#nodes, #degree, node_size)
                    node_neighbor_repr = node_repr[node_neighbor_list, ...]
                    # (#nodes, #degree, edge_size)
                    edge_neighbor_repr = edge_repr[edge_neighbor_list, ...]
                    # (#nodes, #degree, node_size + edge_size)
                    stacked = torch.cat([node_neighbor_repr.float(), edge_neighbor_repr.float()], dim=2)
                    summed = torch.sum(stacked, dim=1, keepdim=False)
                    ## summed = Adjacency matrix * H
                    degree_activation = degree_layer(summed.float())
                    ## degree activation = (Adjacency matrix * H) * W
                    degree_activation_list.append(degree_activation.float())
        neighbor_repr = torch.cat(degree_activation_list, dim=0)
        self_repr = self.linear(node_repr.float())
        # size = (#nodes, #output_size)
        activations = self_repr + neighbor_repr + self.bias.expand_as(self_repr)
        # activations = (Adjacency matrix +  I) * H
        if self.batch_normalize:
            activations = self.normalize(activations.float())
        return F.relu(activations)
