import os.path as osp
import os
import sys
import time
from math import ceil
import numpy as np
from numpy import genfromtxt,  bincount
import sklearn.metrics
from scipy.sparse import base
from sklearn.metrics import cluster, silhouette_score, silhouette_samples
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torchmetrics.clustering import NormalizedMutualInfoScore
from torchmetrics.classification import MulticlassF1Score

from torch_geometric.datasets import TUDataset, Planetoid, AttributedGraphDataset
from torch_geometric.datasets import HeterophilousGraphDataset 
from torch_geometric.datasets import Twitch, JODIEDataset, KarateClub, NELL
from torch_geometric.datasets import KarateClub, Coauthor, CitationFull
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.logging import init_wandb, log

import metrics
import argparse

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset help')
parser.add_argument('--iteration', type=int, help='iteration help')
parser.add_argument('--clustmod', type=int, help='clustmod help')

# output directory
directory_path = 'output'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

n_epochs = 1000
lr = .001
hidden_chnls=64
dout = 0.5

args = parser.parse_args()
#load dataset
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
        
    elif args.dataset == 'Nell':
        dataset = NELL(root='.')
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


fncom = 'AllResults.csv'
fni = sys.argv[0]
num_clusters = data.num_classes+args.clustmod
algo = fni.split('_',1)[0]
fno = algo+'_'+args.dataset+'_'+str(num_clusters)+'_'+str(args.iteration)+'.csv'
#print(fno)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
data = data.to(device)
data.x = data.x.to_dense()
data.x = data.x.to(torch.float32)
adj = to_dense_adj(data.edge_index)

# Run kmeans from sklearn
X = data.x
times = []
start = time.time()
kmeans = KMeans(n_clusters=data.num_classes,
                random_state=None, n_init="auto").fit(X)
times.append(time.time() - start)

# Print metrics
clusters = kmeans.labels_
met_cond = metrics.conductance(adj, clusters)
met_mod = metrics.modularity(adj, clusters)
met_nmi = sklearn.metrics.normalized_mutual_info_score(
      data.y, clusters, average_method='arithmetic')
precision = metrics.pairwise_precision(data.y, clusters)
recall = metrics.pairwise_recall(data.y, clusters)
met_F1 = 2 * precision * recall / (precision + recall)
# if only one cluster predicted, set silhouette manually to 0
if len(np.unique(clusters))==1:
    silsc = 0
else:
    silsc = silhouette_score(data.x,clusters)
log(Epoch='001', Cond = met_cond, Mod = met_mod, NMI = met_nmi, F1 = met_F1, Sil = silsc)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
# write metrics
fno_path = directory_path + '/' + fno
fncom_path = directory_path + '/' + fncom
# write to individual file
with open(fno_path, "w") as file:
    file.write(algo + ',' + args.dataset + ',' + str(num_clusters) + ',' + str(args.iteration) + ',' +
               str(met_cond) + ',' + str(met_mod) + ',' + str(met_nmi) + ',' +
               str(met_F1) + ',' + str(silsc) + ',' + str(torch.tensor(times).median().item()) + '\n')
    file.close()
# append to communal file
with open(fncom_path, "a") as file:
    file.write(algo + ',' + args.dataset + ',' + str(num_clusters) + ',' + str(args.iteration) + ',' +
               str(met_cond) + ',' + str(met_mod) + ',' + str(met_nmi) + ',' +
               str(met_F1) + ',' + str(silsc) + ',' + str(torch.tensor(times).median().item()) + '\n')
    file.close()
