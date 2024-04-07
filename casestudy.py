from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
from sklearn.metrics import f1_score
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE
import scipy.sparse as sp

from torch.nn import Linear
from sklearn.linear_model import LinearRegression

import os
import argparse
import math
import numpy as np
import pandas as pd

from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import argparse



class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(input_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, output_dim)

    def forward(self, x):
        x = F.normalize(x)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    

class MLPM(MessagePassing):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super(MLPM, self).__init__(aggr='add')
        torch.manual_seed(12345)
        self.lin1 = Linear(input_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, output_dim)

    def forward(self, x):
        x = F.normalize(x)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def loss(self, x, y, norm=None, edge_index=None):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        sign =y[edge_index[0]]==y[edge_index[1]]
        sign = [1 if i ==True else -1 for i in sign]
        sign = torch.LongTensor(sign)

        norm = norm[edge_index[0]]
        norm[edge_index[0]==edge_index[1]] = 1

        norm = (norm[edge_index[0]]*norm[edge_index[1]]).pow(0.5)
        x = F.normalize(x)
        x = self.propagate(edge_index, x=x, sign=sign, w=norm)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        #loss = criterion(x, y.view(-1))  # Compute the loss solely based on the training nodes.
              
        return x

    def message(self, x_j, sign, w):
        return sign.view(-1,1)*w.view(-1,1)*x_j


def performance_lr(X, y):
	cls = LinearRegression()
	cls.fit(X,y)
	with open('data/log.txt','a') as f:
		f.write('lr score = '+ str(cls.score(X,y))+'\n')
	print('lr score = ',cls.score(X,y))

def performance_MLPM(X, y, edges):
	# MLPM

	G = nx.from_edgelist(edges.numpy())
	eigen_g = nx.eigenvector_centrality(G, max_iter=10000)
	
	eigen = [eigen_g[i] for i in range(X.shape[0])]
	eigen = torch.FloatTensor(eigen).view(-1)

	model = MLPM(X.shape[1], 16, len(y.unique()))

	criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

	for i in range(200):
	        optimizer.zero_grad()
	        out = model.loss(X, y, eigen, edge_index=edges.T)
	        loss = criterion(out, y.view(-1))  # Compute the loss solely based on the training nodes.
        
	        #print('loss',loss)
	        loss.backward()  
	        optimizer.step()
	        
	out = model(X)
	pred = out.argmax(dim=1)
	correct = pred==y.view(-1)
	accuracy = correct.sum()/correct.shape[0]
	with open('data/log.txt','a') as f:
		f.write('MLPM score = '+ str(accuracy.numpy())+'\n')
	print('MLPM score = ', accuracy.numpy())


def performance_MLP(X, y):
	# MLP

	model = MLP(input_dim=X.shape[1], hidden_channels=16, output_dim = 2)

	criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

	def train():
	    model.train()
	    optimizer.zero_grad()  # Clear gradients.
	    out = model(X)  # Perform a single forward pass.
	    loss = criterion(out, y)  # Compute the loss solely based on the training nodes.
	    loss.backward()  # Derive gradients.
	    optimizer.step()  # Update parameters based on gradients.
	    return loss

	def test():
	    model.eval()
	    out = model(X)
	    pred = out.argmax(dim=1)  # Use the class with highest probability.    
	   
	    test_correct = pred == y.view(-1)  # Check against ground-truth labels.
	    test_acc = int(test_correct.sum()) / len(test_correct)  # Derive ratio of correct predictions.
	    return test_acc

	for epoch in range(1, 201):
	    loss = train()
	    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
	accuracy = test()
	with open('data/log.txt','a') as f:
		f.write('MLP score = '+ str(accuracy)+'\n')
	print('MLP score=', accuracy)


def homo_network(num_nodes):
	adj = np.zeros((num_nodes, num_nodes))
	for i in range(num_nodes//2):
	    for j in range(i+1, num_nodes//2):
	        z = np.random.randint(0, 10, dtype=int)
	        if z > 8:  #0.03
	            adj[i, j] = 1
	            adj[j, i] = 1
	            
	for i in range(num_nodes//2,num_nodes):
	    for j in range(i+1, num_nodes):
	        z = np.random.randint(0, 10, dtype=int)
	        if z > 8:  #0.03
	            adj[i, j] = 1
	            adj[j, i] = 1


	edge_index = []
	x, y = np.where(adj > 0)
	for i in range(len(x)):
	    if x[i] != y[i]:
	        edge_index.append([x[i], y[i]])
	return edge_index

def random_network(num_nodes):
	adj = np.zeros((num_nodes, num_nodes))
	for i in range(num_nodes):
	    for j in range(i+1, num_nodes):
	        z = np.random.randint(0, 10, dtype=int)
	        if z > 8:  #0.03
	            adj[i, j] = 1
	            adj[j, i] = 1
	            
	edge_index = []
	x, y = np.where(adj > 0)
	for i in range(len(x)):
	    if x[i] != y[i]:
	        edge_index.append([x[i], y[i]])
	return edge_index

def hetero_network(num_nodes):

	adj = np.zeros((num_nodes, num_nodes))
	            
	for i in range(num_nodes//2):
	    for j in range(num_nodes//2, num_nodes):
	        z = np.random.randint(0, 10, dtype=int)
	        if z > 8:  #0.03
	            adj[i, j] = 1
	            adj[j, i] = 1

	            
	edge_index = []
	x, y = np.where(adj > 0)
	for i in range(len(x)):
	    if x[i] != y[i]:
	        edge_index.append([x[i], y[i]])
	return edge_index


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_nodes',type=int, default=100)
	parser.add_argument('--epoch', type=int, default=200)
	parser.add_argument('--scale', type=float, default=0.2)
	parser.add_argument('--nx_type', type=str, default='random')
	args = parser.parse_args()
	num_nodes = args.num_nodes
	epoch = args.epoch
	scale = args.scale
	nx_type = args.nx_type

	with open('data/log.txt','a') as f:
		f.write('num_nodes = '+ str(num_nodes)+'\t')
		f.write('scale = '+ str(scale)+'\t')
		f.write('network_type = '+ str(nx_type)+'\n')

	labels = np.array([[0]] * (num_nodes//2) + [[1]] * (num_nodes//2)).astype(np.float32)

	centers = np.array([[0, 1]] * (num_nodes//2) + [[1, 0]] * (num_nodes//2))
	np.random.seed(42)
	dataall = (np.random.normal(0, scale, (num_nodes, 2)) + centers).astype(np.float32)

	if nx_type =='homophily':
		edge_index = homo_network(num_nodes)
	elif nx_type =='random':
		edge_index = random_network(num_nodes)
	elif nx_type =='heterophily':
		edge_index = hetero_network(num_nodes)

	g = nx.from_edgelist(edge_index)
	df = pd.DataFrame(dataall, columns=['d1','d2'])
	df['class'] = labels
	df = df.reset_index()
	df['neighbors'] = df['index'].apply(lambda i: dataall[i] + np.sum([dataall[n] for n in list(nx.neighbors(g, i))]) )


	X, y = torch.FloatTensor(np.array(df[['d1','d2']])), torch.LongTensor(df['class'].tolist())
	edges = torch.LongTensor(edge_index)

	with open('data/log.txt','a') as f:
		f.write('LR score performance 1. all data, 2. weighted data'+'\n')
		

	print('LR score performance 1. all data, 2. weighted data')
	print('1.')
	with open('data/log.txt','a') as f:
		f.write('1.'+'\n')
	performance_lr(X,y)
	performance_MLP(X,y)

	with open('data/log.txt','a') as f:
		f.write('2.'+'\n')
	print('2.')
	performance_MLPM(X,y, edges)
	X, y = torch.FloatTensor(np.array(df['neighbors'].tolist())),torch.LongTensor(np.array(df['class'].tolist()))
	performance_lr(X,y)
	performance_MLP(X,y)






