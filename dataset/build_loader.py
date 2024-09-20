import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler

import pickle
import numpy as np
import random

from dataset.IEMOCAP import IEMOCAP

path = './dataset/IEMOCAP/iemocap4char.pkl'
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

generator = torch.Generator()
generator.manual_seed(seed)

class FixedRandomSampler(Sampler):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        
    def __iter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        n = len(self.data_source)
        rand_perm = torch.randperm(n, generator=self.g).tolist()
        print(f"First five random indices: {rand_perm[:5]}")
        
        return iter(rand_perm)
    
    def __len__(self):
        return len(self.data_source)
    

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataset(data_path=path, id_fold=1, batch_size=8):
    with open(data_path, 'rb') as f:
        IEMOCAP_DataMap = pickle.load(f)
    
    dataset_train, dataset_test, sampler_train = IEMOCAP.Partition(id_fold, IEMOCAP_DataMap, batch_size=8)

    return dataset_train, dataset_test, sampler_train

def build_loader(batch_size, data_path=path, id_fold=1, use_sampler=False):
    dataset_train, dataset_test, sampler_train = build_dataset(data_path=data_path, id_fold=id_fold, batch_size=batch_size)
    if use_sampler:
        print('Emo Sampler!')
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=dataset_train.collate_fn, sampler=sampler_train, shuffle=False)
    else:
        print('FixedRandom Sampler!')
        sampler = FixedRandomSampler(dataset_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=dataset_train.collate_fn, 
                                      sampler=sampler, num_workers=0)
    
    dataloader_test = DataLoader(dataset_test, batch_size=32, collate_fn=dataset_test.collate_fn)
    return dataloader_train, dataloader_test