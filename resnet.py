import torch
import torch.nn as nn
import math
import pickle
import torchvision
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=8631, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class resnet50(nn.Module):    
    def __init__(self, resnet50_path):
        super(resnet50, self).__init__()
        res50 = ResNet(Bottleneck, [3, 4, 6, 3])
        with open(resnet50_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        res50.load_state_dict(weights)
        self.features = nn.Sequential(*list(res50.children())[:-2])  
        self.features2 = nn.Sequential(*list(res50.children())[-2:-1]) 
        # self.features = nn.Sequential(*list(res50.children())[:-1])
        self.classifier = nn.Linear(2048, 7)  
        
    def forward(self, x):
        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        # out=self.classifier(x)
        return x

class ResNet_18(nn.Module):
    def __init__(self, path):
        super(ResNet_18, self).__init__()

        ResNet18 = torchvision.models.resnet18(pretrained=False)
        
        checkpoint = torch.load(path)
        ResNet18.load_state_dict(checkpoint['state_dict'], strict=True)

        self.base = nn.Sequential(*list(ResNet18.children())[:-2])

        self.output = nn.Sequential(nn.Dropout(0.5), Flatten())
        self.classifier = nn.Linear(512, 7)


    def forward(self, image):
        feature_map = self.base(image)
        feature_map = F.avg_pool2d(feature_map, feature_map.size()[2:])
        feature = self.output(feature_map)
        feature = F.normalize(feature, dim=1)


        return  feature
    
def resModel(ResNet18_path, device): #resnet18
    
    model = resnet18(end2end= False,  pretrained= False, num_class=8).to(device)
    
    if  ResNet18_path:
       
        checkpoint = torch.load(ResNet18_path, map_location=device)
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()

        for key in pretrained_state_dict:
            if  ((key == 'fc.weight') | (key=='fc.bias') | (key=='feature.weight') | (key=='feature.bias') ) :
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict, strict = False)
        print('Model loaded from Msceleb pretrained')
    else:
        print('No pretrained resent18 model built.')
    return model   

class EACresnet50(nn.Module):    
    def __init__(self, resnet50_path):
        super(EACresnet50, self).__init__()
        res50 = ResNet(Bottleneck, [3, 4, 6, 3])
        checkpoint = torch.load(resnet50_path)
        res50.load_state_dict(checkpoint['model1_state_dict'])
        self.features = nn.Sequential(*list(res50.children())[:-1])
        self.fc = nn.Linear(2048, 7) 
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    
def resnet18(pretrained=False, num_class=7, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2(BasicBlock, [2, 2, 2, 2], num_class, **kwargs)

    return model

class ResNet2(nn.Module):

    def __init__(self, block, layers, num_classes=7, end2end=True):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(512, 128)
        self.classifier =  nn.Linear(512,num_classes)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
       
        bs = x.size(0)
        f = x

        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        
        f = self.layer1(f)
        
        f = self.layer2(f)
        
        f = self.layer3(f)
        
        f = self.layer4(f)
        
        f = self.avgpool(f)
        
        f = f.squeeze(3).squeeze(2)
        # out_feat=self.head(f)
        pred = self.classifier(f)

        return f;