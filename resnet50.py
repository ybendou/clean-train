import torch
import torch.nn as  nn
import torch.nn.functional as F


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
    def __init__(self, block, num_blocks, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.rotations = rotations
        
        self.conv1 = nn.Conv2d(input_shape[0], feature_maps, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, num_blocks[0], planes=feature_maps)
        self.layer2 = self._make_layer(block, num_blocks[1], planes=feature_maps*2, stride=2)
        self.layer3 = self._make_layer(block, num_blocks[2], planes=feature_maps*4, stride=2)
        self.layer4 = self._make_layer(block, num_blocks[3], planes=feature_maps*8, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(feature_maps*8*block.expansion, num_classes)
        if rotations:   
            self.linear_rot = nn.Linear(feature_maps*8*block.expansion, 4)
        
    def forward(self, x, index_mixup=None, lam=-1):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        out = self.fc(features)
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
        
def ResNet50(feature_maps, input_shape, num_classes, few_shot, rotations, default_torch=False):
    return ResNet(Bottleneck, [3,4,6,3], feature_maps, input_shape, num_classes, few_shot, rotations)