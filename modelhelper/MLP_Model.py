import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, h_sizes, dropout=0.5, activation=F.sigmoid, batch_norm=False):

        # self.loss = torch.nn.MSELoss()
        super(Net, self).__init__()
        self.activation = activation
        self.h_sizes = h_sizes
        self.num_layers = len(self.h_sizes)
        self.linear = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        for i in range(1,self.num_layers):
            self.linear.append(nn.Linear(self.h_sizes[i-1], self.h_sizes[i]))
            
        if self.batch_norm:
            self.batch = nn.ModuleList()
            for j in range(self.num_layers - 1):
                self.batch.append(nn.BatchNorm1d(self.h_sizes[j], affine=False))


    def forward(self, x):
        activation = self.activation
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                x = F.softmax(self.linear[i](x), dim=1)
            else:
                if self.batch_norm:
                    x = self.dropout(activation(self.batch[i](self.linear[i](x))))
                else:
                    x = self.dropout(activation((self.linear[i](x))))

        return x

