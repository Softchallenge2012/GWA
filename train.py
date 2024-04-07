from model import *
from data import *
from test import *
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='cora')
    parser.add_argument('--epoch', type=int, default='100')
    args = parser.parse_args()
    dataset_name = args.dataset
    epoch = args.epoch

    data = load(dataset_name)
    g = nx.from_edgelist(torch.vstack([data['edge_index'][0], data['edge_index'][1]]).T.numpy())

    eigen_cent = nx.eigenvector_centrality(g)
    eigen = torch.zeros(data['y'].shape[0])
    for k, v in eigen_cent.items():
        eigen[k] = v
        
        
    # neighbors = [(i,list(nx.neighbors(g, i))) for i in g.nodes()]
    # neighbors = sorted(neighbors, key= lambda v: v[0])
    neighbors = {}
    for i in g.nodes():
        neighbors[i] = list(nx.neighbors(g, i))
        
    x_u = torch.clone(data['x'])
    for i in range(data['x'].shape[0]):
        if i in neighbors:
            for n in neighbors[i]:
                x_u[i] += eigen[n]*data['x'][n] if data['y'][i]==data['y'][n] else eigen[n]*data['x'][n]*(-1)
    data['weighted'] = x_u

    model = GCNConv(data['x'].shape[1], len(data['y'].unique())) #GCNConv(16, 32)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.01)


    for i in range(epoch):
        
        optimizer.zero_grad()
        out = model.loss(data['x'], data['y'], eigen, edge_index=data['edge_index'])
        loss = criterion(out, data['y'])  # Compute the loss solely based on the training nodes.
        print('loss',loss)
        

        # with torch.no_grad():
        #     valid_loss = test(model, data)

        # early_stopping(loss, valid_loss)
        # if early_stopping.early_stop:
        #     print('We are early stopped at epoch: ', i)
        #     break

        loss.backward()  
        optimizer.step()


    print("accuracy = ", test(model,dataset_name, data))
    
