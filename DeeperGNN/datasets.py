import os.path as osp
import random
import torch
import pandas as pd
from collections import namedtuple

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_coauthor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Coauthor(path, name)
    
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Amazon(path, name)
    
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def get_synthetic_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset_id = random.randint(0,9)

    edge_index = torch.tensor((pd.read_csv('{}/adj'.format(path), sep='\t', header=None)-1).to_numpy().T)
    features = torch.tensor(pd.read_csv('{}/coord'.format(path), sep='\t', header=None).to_numpy(), dtype=torch.float32)
    labels = torch.tensor(pd.read_csv('{}/label'.format(path), sep='\t', header=None).to_numpy(), dtype=torch.int64)[:,dataset_id]
    labels[labels == -1] = 0

    data = Data(x=features, y=labels, edge_index=edge_index)

    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform

    num_nodes, num_features = data.x.shape
    num_classes = data.y.unique().shape[0]
    DataSet = namedtuple('DataSet', ['data', 'num_nodes', 'num_features', 'num_classes'])

    return DataSet(data, num_nodes, num_features, num_classes) if (transform is None) else DataSet(transform(data), num_nodes, num_features, num_classes)



def get_county_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    dat = pd.read_csv('{}/dat.csv'.format(path))
    adj = pd.read_csv('{}/adj.csv'.format(path))

    x = torch.tensor(dat.values[:, 0:9], dtype=torch.float32)
    x = (x - x.mean(dim=0)) / x.std(dim=0)
    y = torch.tensor(dat.values[:, 9] < dat.values[:, 10], dtype=torch.int64)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)

    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform

    num_nodes, num_features = data.x.shape
    num_classes = data.y.unique().shape[0]
    DataSet = namedtuple('DataSet', ['data', 'num_nodes', 'num_features', 'num_classes'])

    return DataSet(data, num_nodes, num_features, num_classes) if (transform is None) else DataSet(transform(data), num_nodes, num_features, num_classes)


def get_sexual_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    dat = pd.read_csv('{}/dat.csv'.format(path), header=None)
    adj = pd.read_csv('{}/adj.csv'.format(path), header=None)

    y = torch.tensor(dat.values[:, 0], dtype=torch.int64)
    x = torch.tensor(dat.values[:, 1:21], dtype=torch.float32)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)

    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform

    num_nodes, num_features = data.x.shape
    num_classes = data.y.unique().shape[0]
    DataSet = namedtuple('DataSet', ['data', 'num_nodes', 'num_features', 'num_classes'])

    return DataSet(data, num_nodes, num_features, num_classes) if (transform is None) else DataSet(transform(data), num_nodes, num_features, num_classes)
