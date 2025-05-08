import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src import setting
from torch.optim.lr_scheduler import ExponentialLR


class DrugsCombModel(nn.Module):

    def __init__(self, drug_a_features_len, drug_b_features_len, cl_features_len):
        super(DrugsCombModel, self).__init__()

        self.drug_a_features_len = drug_a_features_len
        self.drug_b_features_len = drug_b_features_len
        self.cl_features_len = cl_features_len
        self.input_len = drug_a_features_len + drug_b_features_len + cl_features_len

        # Define layers here
        self.layers = self._create_layers()

    def _create_layers(self):
        layers = []
        nodes_nums = setting.FC_layout
        prev_nodes = self.input_len

        for i, nodes in enumerate(nodes_nums):
            layers.append(nn.Linear(prev_nodes, nodes))
            layers.append(nn.BatchNorm1d(nodes))
            layers.append(getattr(nn, setting.activation_method[0])())
            layers.append(nn.Dropout(setting.dropout[i % len(setting.dropout)]))
            prev_nodes = nodes

        layers.append(nn.Linear(prev_nodes, 1))  # Final output layer
        return nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def correlation_coefficient_loss(y_true, y_pred):
        xm = y_true - y_true.mean()
        ym = y_pred - y_pred.mean()
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
        r = r_num / r_den
        r = torch.clamp(r, -1.0, 1.0)
        return 1 - r ** 2

    @classmethod
    def get_loss(cls):
        # Select the loss function
        if setting.loss == 'pearson_correlation':
            return cls.correlation_coefficient_loss
        else:
            return nn.MSELoss()

    @classmethod
    def compile_transfer_learning_model(cls, model):
        # Learning rate scheduler and optimizer
        optimizer = optim.Adam(model.parameters(), lr=setting.start_lr)
        scheduler = ExponentialLR(optimizer, gamma=setting.lr_decay)
        loss_fn = cls.get_loss()

        return model, optimizer, scheduler, loss_fn

    def get_model(self, method=setting.model_type):
        return self

    # def summary(self, input_size):
            # """
            # Print a summary of the model.
            # input_size: tuple of input tensor size (batch_size, channels, features)
            # """
            # from torchinfo import summary
            # return summary(self, input_size==)