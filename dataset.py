
from torch.utils.data import Dataset,DataLoader
from utils import get_annotations, get_disregard, get_num_label
import os
import pandas as pd
import torch
import numpy as np
import cv2
import math

class PrepareDataset(Dataset):
    def __init__(self, root, transforms):
        super(PrepareDataset, self).__init__()
        self.transforms = transforms
        self.image_path = os.path.join(root,'cropped_aligned2')
        self.video_list = []
        self.image_list = []
        for i in os.listdir(self.image_path):
            for j in os.listdir(os.path.join(self.image_path, i)):
                if j.split('.')[-1] != 'jpg':
                    continue
                self.video_list.append(i)
                self.image_list.append(j)
        self.length = len(self.image_list)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_path, self.video_list[idx], self.image_list[idx]))
        img = img[:, :, ::-1]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.video_list[idx], self.image_list[idx]


class AffWild2_Dataset(Dataset):
    def __init__(self, args, phase):
        super(AffWild2_Dataset, self).__init__()
        annotations = get_annotations(args.task)
        root = args.root
        seq_len = args.seq_len
        phase = 'Train_Set' if phase == 'train' else 'Validation_Set'
        feat_path = [os.path.join(root,'features',x+'.npy') for x in args.feature]

        feat_list = [np.load(x, allow_pickle=True).item() for x in feat_path]




        label_path = os.path.join(root,'Annotations')
        disregard = get_disregard(args.task)
        num_label = get_num_label(args.task)
        self.seq_list = []
        for i in os.listdir(label_path):
            df = pd.read_table(os.path.join(label_path, i), header=None, sep=',')
            labels = np.array(df.iloc[1:,:num_label], dtype=np.float32)
            if args.task == 'EXPR':
                labels = labels.astype(int)
            anno_index = np.argwhere(np.sum(labels == disregard, axis=1) == 0).flatten()
            feat_index = np.array([int(x[:-4]) - 1 for x in feat_list[0][i[:-4]].keys()])
            index = sorted(set(anno_index).intersection(set(feat_index)))
            num_idx = len(index)
            num_seq = math.ceil(num_idx / seq_len)
            for idx in range(num_seq):
                seq_idx = index[idx * seq_len: min((idx + 1) * seq_len, num_idx)]
                seq_feat = np.concatenate([np.array([y[i[:-4]]['{}.jpg'.format(str(x + 1).zfill(5))] for x in seq_idx]) for y in feat_list],axis=1)
                seq_label = labels[seq_idx,:]
                if seq_feat.shape[0] < seq_len:
                    seq_feat = np.pad(seq_feat, ((0, seq_len - seq_feat.shape[0]), (0, 0)), 'edge')
                    seq_label = np.pad(seq_label, ((0, seq_len - seq_label.shape[0]), (0, 0)), 'edge')
                self.seq_list.append({'feat':seq_feat, 'label':seq_label})
        self.length = len(self.seq_list)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.seq_list[idx]['feat']), torch.from_numpy(self.seq_list[idx]['label']).squeeze()
    
    def get_feature_dim(self):
        return self.seq_list[0]['feat'].shape[1]


class testDataset(Dataset):
    def __init__(self, path = '/data/shenkang/data/ABAW', feats_paths = None):
        self.path = path

        if feats_paths is not None:
            self.feats = []
            for feat_path in feats_paths:
                self.feats.append(np.load(feat_path,allow_pickle=True).item())
 
        with open(os.path.join(path, 'MTL.txt'),'r')as f:
            lines = f.readlines()

        self.labels = [line.replace('\n','').split(',')[0] for line in lines[1:]]
        # self.labels.remove('40-30-1280x720/06926.jpg')
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        first,second = self.labels[idx].split('/')
        features = []
        for i in self.feats:
            features.append(torch.tensor(i[first][second]))
 
        return features,first+'/'+second


'''
.npy文件为二级字典
对应文件夹名和文件名
'''

if __name__ == '__main__':
    files = ['23_poster2_affect8.npy','ASM_resnet18.npy',"23_fau.npy","POSTER2_aff8.npy","23_resnet18.npy"]
    dataset = testDataset(feats_paths=[os.path.join('/data/shenkang/data/ABAW/test_features', file) for file in files])

    testloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    for batch, (features) in enumerate(testloader):
        pass