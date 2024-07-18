import torch.utils.data as data
import pandas as pd
import os, torch
import cv2
from torchvision import transforms
import numpy as np

from CEF.aug import aug1,aug2,aug3


class Dataset(data.Dataset):
    def __init__(self, path = '/data/shenkang/data/ABAW',add_dataset = ['aff','c','raf'], phase = 'train'):

        self.phase = phase
        self.add_dataset = add_dataset
        self.path = path
        self.aug1 = aug1()
        self.aug2 = aug2()
        self.aug3 = aug3()
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        
        self.label = {}
        if(self.phase == 'train'):
            with open(os.path.join(path,'compound','EmoLabel','list_patition_label.txt'),'r')as f:
                lines = f.readlines()
                for line in lines:
                    data = line.split(' ')
                    if(phase in data[0]):
                        if int(data[1]) == 1:
                            anno = 1
                        elif(int(data[1]) == 3):
                            anno = 5
                        elif(int(data[1]) == 4):
                            anno = 6
                        elif(int(data[1]) == 5):
                            anno = 2
                        elif(int(data[1]) == 8):
                            anno = 0
                        elif(int(data[1]) == 9):
                            anno = 4
                        elif(int(data[1]) == 11):
                            anno = 3
                        else:
                            anno = -1
                        self.label[data[0]] = anno
            
            if 'c' in add_dataset:
                first = os.listdir(os.path.join('/data/liuxuxiong/data','video2frame'))
                for i in first:
                    files = os.listdir(os.path.join('/data/liuxuxiong/data','video2frame',i))
                    for file in files:
                        self.label['{}/{}'.format(i,file)] = -1
                        
            if 'aff' in add_dataset:
                with open(os.path.join(path,'Annotations','training_set_annotations.txt'),'r') as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        data = line.split(',')
                        if(data[3] != '-1' and data[3] != '0' and data[3] != '7'):
                            self.label[data[0] + 'aff'] = int(data[3]) - 1
                
                with open(os.path.join(path,'Annotations','validation_set_annotations.txt'),'r') as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        data = line.split(',')
                        if(data[3] != '-1' and data[3] != '0' and data[3] != '7'):
                            # aff标签
                            # Neutral   Anger   Disgust Fear    Happiness   Sadness Surprise    Other
                            # 0 1 2 3 4 5 6 7
                            # 过滤后原始标签是1-6
                            self.label[data[0] + 'aff'] = int(data[3]) - 1
                            
            if 'raf' in add_dataset:
                with open(os.path.join(path.replace('ABAW','RAF-DB'),'EmoLabel','train_label.txt'),'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        data = line.replace('\n','').split(' ')
                        # raf标签
                        # Surprise  Fear    Disgust  Happiness   Sadness   Anger   Neutral
                        # 1 2   3   4   5   6   7
                        # 过滤后原始标签是1-6
                        
                        if(data[1] == '1'):
                            self.label[data[0] + 'raf'] = 5
                        elif(data[1] == '2'):
                            self.label[data[0] + 'raf'] = 2
                        elif(data[1] == '3'):
                            self.label[data[0] + 'raf'] = 1
                        elif(data[1] == '4'):
                            self.label[data[0] + 'raf'] = 3
                        elif(data[1] == '5'):
                            self.label[data[0] + 'raf'] = 4
                        elif(data[1] == '6'):
                            self.label[data[0] + 'raf'] = 0
                        
                            
                with open(os.path.join(path.replace('ABAW','RAF-DB'),'EmoLabel','test_label.txt'),'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        data = line.replace('\n','').split(' ')
                        if(data[1] == '1'):
                            self.label[data[0] + 'raf'] = 5
                        elif(data[1] == '2'):
                            self.label[data[0] + 'raf'] = 2
                        elif(data[1] == '3'):
                            self.label[data[0] + 'raf'] = 1
                        elif(data[1] == '4'):
                            self.label[data[0] + 'raf'] = 3
                        elif(data[1] == '5'):
                            self.label[data[0] + 'raf'] = 4
                        elif(data[1] == '6'):
                            self.label[data[0] + 'raf'] = 0
                    
        else:
            with open(os.path.join(path,'compound','EmoLabel','list_patition_label.txt'),'r')as f:
                lines = f.readlines()
                for line in lines:
                    data = line.split(' ')
                    if(phase in data[0]):
                        if int(data[1]) == 1:
                            anno = 1 # Happily Surprised 对应Aff 4+6(原始索引) 相应需要索引减一
                        elif(int(data[1]) == 3):
                            anno = 5 # Sadly Fearful 对应Aff 5+3
                        elif(int(data[1]) == 4):
                            anno = 6 # Sadly Angry 对应Aff 5+1
                        elif(int(data[1]) == 5):
                            anno = 2 # Sadly Surprised 对应Aff 5+6
                        elif(int(data[1]) == 8):
                            anno = 0 # Fearfully Surprised 对应Aff 3+6
                        elif(int(data[1]) == 9):
                            anno = 4 # Angrily Surprised 对应Aff 1+6
                        elif(int(data[1]) == 11):
                            anno = 3 # Disgustedly Surprised 对应Aff 2+6
                        else:
                            anno = -1
                        self.label[data[0]] = anno
                    
    def __len__(self):
        return len(self.label.keys())
    
    def __getitem__(self, index):
        key = list(self.label.keys())[index]
        
        # print(key)
        
        if('/' in key and 'aff' not in key):
            CElabel = -1
            FERlabel = -1
            img = cv2.imread(os.path.join('/data/liuxuxiong/data','video2frame',key))
        elif('raf' in key):
            
            CElabel = -1
            FERlabel = self.label[key]
            img = cv2.imread(os.path.join(self.path.replace('ABAW','RAF-DB'),'Image/aligned/aligned',key.replace('raf','').replace('.jpg','_aligned.jpg')))
        elif('aff' in key):
            CElabel = -1
            FERlabel = self.label[key]
            img = cv2.imread(os.path.join(self.path,'cropped_aligned',key.replace('aff','')))
        else:
            CElabel = self.label[key]
            FERlabel = -1
            img = cv2.imread(os.path.join(self.path,'compound/Image/aligned',key.replace('.jpg','_aligned.jpg')))
        
        img = img[:, :, ::-1]
        img_ori = self.transforms(self.aug3(img.copy()))
        img_1 = self.transforms(self.aug1(img.copy()))
        img_2 = self.transforms(self.aug2(img.copy()))
        
        
        
        return img_ori, img_1, img_2, CElabel, FERlabel
        

class testDataset(data.Dataset):
    def __init__(self):
        
        self.aug3 = aug3()
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        
        # with open('/data/shenkang/data/ABAW/CER.txt','r')as f:
        #     lines = f.readlines()
                
        # self.labels = [line.replace('\n','').split(',')[0] for line in lines[1:]]
        
        self.labels = []
        folders = os.listdir('/data/liuxuxiong/data/video2frame')
        for folder in folders:
            files = os.listdir('/data/liuxuxiong/data/video2frame/' + folder)
            for file in files:
                self.labels.append(folder + '/' + file)
        
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        key = self.labels[index]
        img = cv2.imread(os.path.join('/data/liuxuxiong/data','video2frame',key))
        img = img[:, :, ::-1]
        img_ori = self.transforms(self.aug3(img.copy()))
        return img_ori,key
        
        
if __name__ == '__main__':
    test_dataset = testDataset()
    print(len(test_dataset))
    
    # train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # # dataset = Dataset()
    # for batch, (img_ori, img_1, img_2, label) in enumerate(train_dataloader):
    #     print(batch)
    # print(dataset[0])
        
        
