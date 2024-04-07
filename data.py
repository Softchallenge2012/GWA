import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor, WebKB, WikipediaNetwork


def load(dataset_name):

    path = 'data'

    if dataset_name == 'test':
        edge_index = torch.tensor([[0, 1],
                                   [1, 0],
                                   [1, 2],
                                   [2, 1]], dtype=torch.long)
        x = torch.tensor([[0], [1], [2]], dtype=torch.float)
        y = torch.tensor([0,0,1], dtype=torch.long)

        data = Data(x=x,y=y, edge_index=edge_index.t().contiguous())

    elif dataset_name == str.lower('cora'):
        dataset = Planetoid(path,'cora')
        data = dataset.data
    elif dataset_name == str.lower('citeseer'):
        dataset = Planetoid(path,'citeseer')
        data = dataset.data
    elif dataset_name == str.lower('pubmed'):
        dataset = Planetoid(path,'pubmed')
        data = dataset.data
    elif dataset_name == str.lower('cornell'):
        dataset = WebKB(path,'Cornell')
        data = dataset.data
    elif dataset_name == str.lower('Wisconsin'):
        dataset = WebKB(path,'Wisconsin')
        data = dataset.data
    elif dataset_name == str.lower('Texas'):
        dataset = WebKB(path,'Texas')
        data = dataset.data
    elif dataset_name =='chameleon':
        dataset = WikipediaNetwork(path,'chameleon')
        data = dataset.data
    elif dataset_name =='squirrel':
        dataset = WikipediaNetwork(path,'squirrel')
        data = dataset.data
    if data == None:
        print('data was not loaded')
        return None
    else:
        return data

