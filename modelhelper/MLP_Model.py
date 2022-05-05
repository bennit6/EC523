import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, h_sizes, dropout=0.5, activation=F.sigmoid):

        # self.loss = torch.nn.MSELoss()
        super(Net, self).__init__()
        self.activation = activation
        self.h_sizes = h_sizes
        self.num_layers = len(self.h_sizes)
        self.fc = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(self.num_layers):
            self.fc.append(nn.Linear(self.h_sizes[i][0], self.h_sizes[i][1]))


    def forward(self, x):
        activation = self.activation
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                x = F.softmax(self.fc[i](x), dim=1)
            else:
                x = self.dropout(activation(self.fc[i](x)))
        return x