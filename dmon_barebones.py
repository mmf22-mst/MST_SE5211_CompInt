###############################################################################
# Library imports
import os
import sys
import time
import metrics
import argparse
import numpy as np
from numpy import genfromtxt
import sklearn.metrics
from sklearn.metrics import silhouette_score
import torch
from torch_geometric.datasets import Planetoid, AttributedGraphDataset
from torch_geometric.datasets import HeterophilousGraphDataset 
from torch_geometric.datasets import KarateClub, Coauthor, CitationFull
from torch_geometric.data import Data
from torch_geometric.nn import DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.logging import log

###############################################################################
# Parse command line args and set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset help')
parser.add_argument('--iteration', type=int, help='iteration help')
parser.add_argument('--clustmod', type=int, help='clustmod help')
args = parser.parse_args()
n_epochs = 1000
lr = .001
hidden_chnls=64
dout = 0.5

###############################################################################
# Load dataset
if args.dataset == 'Iris':
    iris = genfromtxt('Iris_proc.csv', delimiter=',')
    nodes = torch.from_numpy(iris[:,0])-1
    nodes = nodes.to(torch.int32)
    x = torch.from_numpy(iris[:,:-1])
    x = x.to(torch.float32)
    y = torch.from_numpy(iris[:,-1])
    y = y.to(torch.int32)
    ds = Data()
    ds.x = x
    ds.y = y
    ds.num_classes = len(np.unique(ds.y))
    ds.edge_index = torch.stack((nodes,nodes),1).transpose(1, 0).to(torch.long)
    dataset = 'Iris()'
    data = ds
elif args.dataset == 'Iris_rand':
    iris = genfromtxt('Iris_proc_rand.csv', delimiter=',')
    nodes = torch.from_numpy(iris[:,0])-1
    nodes = nodes.to(torch.int32)
    x = torch.from_numpy(iris[:,:-1])
    x = x.to(torch.float32)
    y = torch.from_numpy(iris[:,-1])
    y = y.to(torch.int32)
    ds = Data()
    ds.x = x
    ds.y = y
    ds.num_classes = len(np.unique(ds.y))
    ds.edge_index = torch.stack((nodes,nodes),1).transpose(1, 0).to(torch.long)
    dataset = 'Iris()'
    data = ds
else:
    if args.dataset == 'cora':
        dataset = Planetoid(root='.', name='Cora')
    elif args.dataset == 'CF_Cora_ML':
        dataset = CitationFull(root='.', name='Cora_ML')
    elif args.dataset == 'Flickr':
        dataset = AttributedGraphDataset(root='.', name='Flickr')
    elif args.dataset == 'KarateClub':
        dataset = KarateClub()
    elif args.dataset == 'CF_Cora':
        dataset = CitationFull(root='.', name='Cora')
    elif args.dataset == 'CF_PubMed':
        dataset = CitationFull(root='.', name='PubMed')
    elif args.dataset == 'CA_CS':
        dataset = Coauthor(root='.', name='CS')
    elif args.dataset == 'Tolokers':
        dataset = HeterophilousGraphDataset(root='.', name='Tolokers')

    data = dataset[0]
    data.num_classes = dataset.num_classes

###############################################################################
# Output filenames, etc.
directory_path = 'output'
if not os.path.exists(directory_path):
    os.makedirs(directory_path) 
fncom = 'AllResults.csv'
fni = sys.argv[0]
num_clusters = data.num_classes+args.clustmod
algo = fni.split('_',1)[0]
fno = algo+'_'+args.dataset+'_'+str(num_clusters)+'_'+str(args.iteration)+'.csv'
fno_path = directory_path + '/' + fno
fncom_path = directory_path + '/' + fncom

###############################################################################
# PyTorch neural net class
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=hidden_chnls):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = DMoNPooling([hidden_channels,
                                  hidden_channels],
                                  data.num_classes,
                                  dropout=dout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        adj = to_dense_adj(edge_index)
        s, x, adj, sp1, _, c1 = self.pool1(x, adj)

        return s[0], x[0], sp1 + c1

###############################################################################
# Set device (cpu/gpu) and define model and optimizer
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
data = data.to(device)
data.x = data.x.to_dense()
data.x = data.x.to(torch.float32)
model = Net(data.num_features, num_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)

###############################################################################
# Define training loop, forward and backward passes
def train():
    model.train()
    optimizer.zero_grad()
    _, _, tot_loss = model(data.x, data.edge_index)
    loss = tot_loss
    loss.backward()
    optimizer.step()
    return float(loss)

###############################################################################
# Train model for # of epochs
times = []
for epoch in range(1, n_epochs+1):
    start = time.time()
    _ = train()
    times.append(time.time() - start)
model.eval()
s, _, _ = model(data.x, data.edge_index)
clusters = s.argmax(dim=-1)
adj = to_dense_adj(data.edge_index)

###############################################################################
# Calculate and write metrics to stdio and files
met_cond = metrics.conductance(adj, clusters)
met_mod = metrics.modularity(adj, clusters)
met_nmi = sklearn.metrics.normalized_mutual_info_score(
      data.y, clusters, average_method='arithmetic')
precision = metrics.pairwise_precision(data.y, clusters)
recall = metrics.pairwise_recall(data.y, clusters)
met_F1 = 2 * precision * recall / (precision + recall)

# error catch ******************************************************
# if only one cluster predicted, set silhouette manually to 0
if len(np.unique(clusters))==1:
    silsc = 0
else:
    silsc = silhouette_score(data.x,clusters)
# error catch ******************************************************
    
# log automatically outputs to stdio
log(Epoch=epoch, Cond = met_cond, Mod = met_mod, NMI = met_nmi, F1 = met_F1, Sil = silsc)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
# write metrics to files (fno = individual run, fncom = all runs appended)
with open(fno_path, "w") as file:
    file.write(algo + ',' + args.dataset + ',' + str(num_clusters) + ',' + str(args.iteration) + ',' +
               str(met_cond) + ',' + str(met_mod) + ',' + str(met_nmi) + ',' +
               str(met_F1) + ',' + str(silsc) + ',' + str(torch.tensor(times).median().item()) + '\n')
    file.close()
with open(fncom_path, "a") as file:
    file.write(algo + ',' + args.dataset + ',' + str(num_clusters) + ',' + str(args.iteration) + ',' +
               str(met_cond) + ',' + str(met_mod) + ',' + str(met_nmi) + ',' +
               str(met_F1) + ',' + str(silsc) + ',' + str(torch.tensor(times).median().item()) + '\n')
    file.close()
