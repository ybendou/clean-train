import torch
import torch.nn as  nn
import torch.nn.functional as F
from args import * 
from utils import linear
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            x = self.shortcut(x)
        out += x
        if args.dropout > 0:
            out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.bn3(x)
        
        #downsample if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, input_shape, num_classes, few_shot, rotations, min_size=100):
        super(ResNet, self).__init__()
        self.in_channels = feature_maps
        self.rotations = rotations
        self.input_shape = input_shape
        self.min_size = min_size
        if input_shape[1]>self.min_size:
            self.conv1 = nn.Conv2d(input_shape[0], feature_maps, kernel_size=7, stride=2, padding=3, bias=False) # kernel size 7 instead of 3 
        else:
            self.conv1 = nn.Conv2d(input_shape[0], feature_maps, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(feature_maps)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, num_blocks[i], planes=feature_maps*(2**i), stride=1 if i==0 else 2))
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = linear(feature_maps*(2**i)*block.expansion, num_classes)
        if rotations:   
            self.linear_rot = linear(feature_maps*(2**i)*block.expansion, 4)
        
    def forward(self, x, index_mixup=None, lam=-1):
        if lam!= -1:
            mixup_layer = random.randint(0, len(self.layers))
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        out = self.relu(self.bn1(self.conv1(out)))
        if self.input_shape[1]>self.min_size:
            out = self.max_pool(out)
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i+1:
                out = lam * out + (1-lam) * out[index_mixup]
                out = F.relu(out) # made it only for mixup (there is already a relu before inside the block)
        out = self.avgpool(out)
        features = out.view(out.shape[0], -1)
        out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features
        
    def _make_layer(self, block, blocks, planes, stride=1):
        downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
            
        layers.append(block(self.in_channels, planes, downsample=downsample, stride=stride))
        self.in_channels = planes*block.expansion
        
        for i in range(blocks-1):
            layers.append(block(self.in_channels, planes))
            
        return nn.Sequential(*layers)
        

def ResNet18(feature_maps, input_shape, num_classes, few_shot, rotations):
    return ResNet(BasicBlock, [2, 2, 2, 2], feature_maps, input_shape, num_classes, few_shot, rotations)

def ResNet20(feature_maps, input_shape, num_classes, few_shot, rotations):
    return ResNet(BasicBlock, [3, 3, 3], feature_maps, input_shape, num_classes, few_shot, rotations)

def ResNet50(feature_maps, input_shape, num_classes, few_shot, rotations):
    return ResNet(Bottleneck, [3,4,6,3], feature_maps, input_shape, num_classes, few_shot, rotations)

def ResNet56(feature_maps, input_shape, num_classes, few_shot, rotations):
    return ResNet(BasicBlock, [9, 9, 9], feature_maps, input_shape, num_classes, few_shot, rotations)

def ResNet110(feature_maps, input_shape, num_classes, few_shot, rotations):
    return ResNet(BasicBlock, [18, 18, 18], feature_maps, input_shape, num_classes, few_shot, rotations)

def ResNet1202(feature_maps, input_shape, num_classes, few_shot, rotations):
    return ResNet(BasicBlock, [200, 200, 200], feature_maps, input_shape, num_classes, few_shot, rotations)

