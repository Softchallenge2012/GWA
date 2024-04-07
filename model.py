import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops

from time import perf_counter as t
import yaml
from yaml import SafeLoader
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Actor, WebKB
from torch_geometric.utils import dropout_adj, subgraph
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj, homophily
import networkx as nx
import matplotlib.pyplot as plt
#from model import Model, LogReg
#from utils import knn_graph


import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx

from torch.nn.modules.module import Module


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        torch.manual_seed(12345)
        self.lin1 = Linear(in_channels, in_channels*2)
        self.lin2 = Linear(in_channels*2, out_channels)


        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        #self.bias.data.zero_()
        
    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    

    def loss(self, x, y, norm, edge_index=None):
       

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        sign =y[edge_index[0]]==y[edge_index[1]]
        sign = [1 if i ==True else -1 for i in sign]
        sign = torch.LongTensor(sign)
        
        norm = norm[edge_index[0]]
        norm[edge_index[0]==edge_index[1]] = 1

        x = self.propagate(edge_index, x=x, sign=sign, w=norm)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin2(x)
        
        
        return out

    def message(self, x_j, sign, w):
        return sign.view(-1,1)*w.view(-1,1)*x_j #.view(-1,1)

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

