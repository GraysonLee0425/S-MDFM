import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """ 单时期 ResNet backbone, 提取时序特征 """
    def __init__(self, block, blocks_num, groups=1, width_per_group=64):
        super(ResNetBackbone, self).__init__()
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(9, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)  # 单时期9通道
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 (B, 512*expansion)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # (B,9,H,W) -> (B,64,H/2,W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # (B,64,H/4,W/4)
        x = self.layer1(x)    # (B,64,*,*)
        x = self.layer2(x)    # (B,128,*,*)
        x = self.layer3(x)    # (B,256,*,*)
        x = self.layer4(x)    # (B,512,*,*)
        x = self.avgpool(x)   # (B,512,1,1)
        x = torch.flatten(x, 1)  # (B,512)
        return x


class ResNetMultiPeriod(nn.Module):
    """ 多时期 + 注意力融合 + 文本特征 """
    def __init__(self, block, blocks_num, num_classes=1, text_feature_dim=18):
        super(ResNetMultiPeriod, self).__init__()
        self.period_backbone = ResNetBackbone(block, blocks_num)  # 共享权重
        self.num_periods = 4

        feat_dim = 512 * block.expansion
        self.attention = nn.Linear(feat_dim, 1)  # 注意力权重计算
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(128 + text_feature_dim, num_classes)

    def forward(self, x_list, text_features):
        """
        x_list: list[Tensor], 每个元素是 (B, 9, H, W)，共4个时期
        text_features: (B, text_feature_dim)
        """
        feats = []
        for t in range(self.num_periods):
            feat_t = self.period_backbone(x_list[t])  # (B, 512)
            feats.append(feat_t)
        feats = torch.stack(feats, dim=1)  # (B,4,512)

        # 注意力融合
        attn_score = self.attention(feats)  # (B,4,1)
        attn_weight = torch.softmax(attn_score, dim=1)  # (B,4,1)
        fused_feat = torch.sum(feats * attn_weight, dim=1)  # (B,512)

        # 全连接 + 文本特征
        fused_feat = self.fc1(fused_feat)  # (B,128)
        out = torch.cat([fused_feat, text_features], dim=1)
        out = self.fc2(out)  # (B,num_classes)
        return out


def resnet18_multip(num_classes=1, text_feature_dim=18):
    return ResNetMultiPeriod(BasicBlock, [2, 2, 2, 2],
                             num_classes=num_classes,
                             text_feature_dim=text_feature_dim)


def resnet50_multip(num_classes=1, text_feature_dim=18):
    return ResNetMultiPeriod(Bottleneck, [3, 4, 6, 3],
                             num_classes=num_classes,
                             text_feature_dim=text_feature_dim)
