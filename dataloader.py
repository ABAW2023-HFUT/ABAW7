import torch.utils.data as data
import pandas as pd
import os, torch
import cv2
from torchvision import transforms
import numpy as np

from augment import *

class Dataset(data.Dataset):
    def __init__(self, path = '/data/shenkang/data/ABAW/Annotations', image_path = 'cropped_aligned', phase = 'train', transform = None, feats_paths = None, feat_aug=False):
        self.path = path
        self.phase = phase
        self.image_path = image_path
        self.transform = transform
        self.feat_aug = feat_aug
        
        self.feats_paths = feats_paths
        if self.feats_paths is not None:
            self.feats = []
            for feat_path in self.feats_paths:
                self.feats.append(np.load(feat_path,allow_pickle=True).item())
        else:
            self.feats = None
        
        self.labels = []
        if phase == 'train':
            with open(os.path.join(path, 'training_set_annotations.txt'), 'r')as f:
                lines = f.readlines()
        if phase == 'val':
            with open(os.path.join(path, 'validation_set_annotations.txt'), 'r')as f:
                lines = f.readlines()
                
        for line in lines[1:]:
            res = line.replace('\n','').split(',')
            cur = {}
            cur['image'] = res[0]
            cur['valence'] = float(res[1])
            cur['arousal'] = float(res[2])
            cur['expression'] = int(res[3])
            cur['aus'] = [int(i) for i in res[4:]]
            self.labels.append(cur)
            
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        if(self.feats == None):
            image = cv2.imread(os.path.join(self.path.replace('/Annotations',''), self.image_path, self.labels[idx]['image']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            return image,\
                    torch.tensor(self.labels[idx]['valence'], dtype = torch.float32),\
                    torch.tensor(self.labels[idx]['arousal'], dtype =torch.float32),\
                    torch.tensor(self.labels[idx]['expression'], dtype = torch.long),\
                    torch.tensor(self.labels[idx]['aus'], dtype = torch.long)
        else:
            if('/' in self.labels[idx]['image']):
                first,second = self.labels[idx]['image'].split('/')
                features = []
                for i in self.feats:
                    if(self.feat_aug):
                        features.append(feature_perturbation(torch.tensor(i[first][second]),int(0.5 * i[first][second].shape[0]), 0.1))
                    else:
                        features.append(torch.tensor(i[first][second]))
                
                return features,\
                    torch.tensor(self.labels[idx]['valence'], dtype = torch.float32),\
                    torch.tensor(self.labels[idx]['arousal'], dtype =torch.float32),\
                    torch.tensor(self.labels[idx]['expression'], dtype = torch.long),\
                    torch.tensor(self.labels[idx]['aus'], dtype = torch.long)
   
                
'''
.npy文件为二级字典
对应文件夹名和文件名
'''                
                               
if __name__ == '__main__':
    # # transform = transforms.Compose([
    # #     transforms.ToTensor(),
    # #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    # #                          std=[0.229, 0.224, 0.225])
    # # ])
    # transform = None
    # files = os.listdir('/data/shenkang/ABAW/feature2023/Aff-Wild2/features')
    # # 读特征就不需要image_path,transform
    # files = ['resnet18.npy']
    # dataset = Dataset(path = '/data/shenkang/data/ABAW/Annotations', image_path = 'cropped_aligned', phase = 'train', transform=transform, feats_paths = [os.path.join('/data/shenkang/ABAW/feature2023/Aff-Wild2/features',file) for file in files])
    # dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # # for batch,(image, valence, arousal, expression, aus) in enumerate(dataloader):
    #     # pass
    # for batch,(features, valence, arousal, expression, aus) in enumerate(dataloader):
    #     # features 是 list, list每个元素是[batchsize, channel]的torch.tensor
    #     pass
    #     # for i in features:
    #     #     print(i.shape)
    #     # break
        
    data = np.load('/data/shenkang/ABAW/feature2023/Aff-Wild2/features/eac.npy',allow_pickle=True).item()
    
    new = {}
    
