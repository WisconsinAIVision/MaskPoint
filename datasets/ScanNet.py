import os
import json
import torch
import numpy as np
import torch.utils.data as data

from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class ScanNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.folder = config.FOLDER
        self.npoints = int(config.N_POINTS)
        self.split = config.SPLIT
        with open(os.path.join(self.data_root, f'catalog_{self.split}.json')) as fp:
            self.data_objs = json.load(fp)
        print_log(f'[DATASET] {len(self.data_objs)} instances loaded from {self.split} split.', logger = 'ScanNet')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def sample_pts(self, point_cloud, num):
        if len(point_cloud) >= num:
            idx = np.random.choice(len(point_cloud), num, replace=False)
        else:
            idx = np.random.choice(len(point_cloud), num, replace=True)
        return point_cloud[idx,:]

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_root, self.folder, self.data_objs[idx]))
        data = data[:, 0:3]

        data = self.sample_pts(data, self.npoints)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return 0, 0, data

    def __len__(self):
        return len(self.data_objs)
