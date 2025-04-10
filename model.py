import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)#批归一化
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 第一个卷积块
        out = F.gelu(self.bn1(self.conv1(x)))
        
        # 第二个卷积块
        out = self.bn2(self.conv2(out))
        
        # 残差连接
        shortcut = self.shortcut(x)
        out = out + shortcut
        
        # 通道注意力
        attention = self.channel_attention(out)
        out = out * attention
        
        # 最大池化
        out = self.pool(out)
        return out

class EnhancedResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedResNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 三个主要卷积块
        self.block1 = ConvBlock(64, 128)
        self.block2 = ConvBlock(128, 256)
        self.block3 = ConvBlock(256, 512)
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),  # Dropout正则化
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # 使用Kaiming初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始卷积
        out = F.gelu(self.bn1(self.conv1(x)))
        
        # 三个卷积块
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        # 自适应池化
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = self.fc(out)
        return out 