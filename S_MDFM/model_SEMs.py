import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
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
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
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

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64, text_feature_dim=18, sems_dim=3, fc1_dim=128, text_proj_dim=32, sems_proj_dim=32):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(9, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.text_feature_dim = text_feature_dim
        self.sems_dim = sems_dim
        self.fc1_dim = fc1_dim
        self.text_proj_dim = text_proj_dim
        self.sems_proj_dim = sems_proj_dim
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(512 * block.expansion, self.fc1_dim)
            self.fc_text = nn.Linear(self.text_feature_dim, self.text_proj_dim)
            self.fc_sems = nn.Linear(self.sems_dim, self.sems_proj_dim)
            self.fc_film = nn.Linear(self.text_proj_dim + self.sems_proj_dim, self.fc1_dim * 2)
            self.fc_out = nn.Sequential(nn.Linear(self.fc1_dim + self.text_proj_dim + self.sems_proj_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, groups=self.groups, width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    @staticmethod
    def aggregate_sems_from_history(train_loss_history, val_rmse_history, val_mae_history, last_k=100, device=None):
        import numpy as _np
        def to_np_array(x):
            if x is None:
                return _np.array([])
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            if isinstance(x, list):
                return _np.array(x)
            return _np.array(x)
        t = to_np_array(train_loss_history)
        r = to_np_array(val_rmse_history)
        m = to_np_array(val_mae_history)
        if t.size == 0:
            t_avg = 0.0
        else:
            t_avg = float(t[-last_k:].mean())
        if r.size == 0:
            r_avg = 0.0
        else:
            r_avg = float(r[-last_k:].mean())
        if m.size == 0:
            m_avg = 0.0
        else:
            m_avg = float(m[-last_k:].mean())
        arr = torch.tensor([t_avg, r_avg, m_avg], dtype=torch.float32)
        if device is not None:
            arr = arr.to(device)
        return arr
    def forward(self, x, text_features, prev_metrics=None):
        device = next(self.parameters()).device
        x = x.to(device)
        text_features = text_features.to(device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            if prev_metrics is None:
                t = self.fc_text(text_features)
                t = torch.relu(t)
                s = torch.zeros(x.size(0), self.sems_proj_dim, device=device)
                gamma = torch.ones(x.size(0), self.fc1_dim, device=device)
                beta = torch.zeros(x.size(0), self.fc1_dim, device=device)
            else:
                if isinstance(prev_metrics, (list, tuple)):
                    pm = torch.tensor(prev_metrics, dtype=torch.float32, device=device)
                    if pm.dim() == 1 and pm.size(0) == self.sems_dim:
                        sems = pm.unsqueeze(0).repeat(x.size(0), 1)
                    else:
                        sems = pm
                elif isinstance(prev_metrics, torch.Tensor):
                    sems = prev_metrics
                    if sems.dim() == 1 and sems.size(0) == self.sems_dim:
                        sems = sems.unsqueeze(0).repeat(x.size(0), 1)
                else:
                    sems = torch.tensor(prev_metrics, dtype=torch.float32, device=device)
                    if sems.dim() == 1 and sems.size(0) == self.sems_dim:
                        sems = sems.unsqueeze(0).repeat(x.size(0), 1)
                if sems.device != device:
                    sems = sems.to(device)
                t = self.fc_text(text_features)
                t = torch.relu(t)
                s = self.fc_sems(sems)
                s = torch.relu(s)
                film_in = torch.cat((t, s), dim=1)
                film_params = self.fc_film(film_in)
                gamma, beta = torch.split(film_params, self.fc1_dim, dim=1)
                gamma = torch.sigmoid(gamma * 1.0) * 2.0
                beta = torch.tanh(beta)
            x = gamma * x + beta
            fused = torch.cat((x, t, s), dim=1)
            out = self.fc_out(fused)
            return out
        return x

def resnet18(num_classes=1000, include_top=True, text_feature_dim=18, sems_dim=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, text_feature_dim=text_feature_dim, sems_dim=sems_dim)

def resnet34(num_classes=1000, include_top=True, text_feature_dim=19, sems_dim=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, text_feature_dim=text_feature_dim, sems_dim=sems_dim)

def resnet50(num_classes=1000, include_top=True, text_feature_dim=19, sems_dim=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, text_feature_dim=text_feature_dim, sems_dim=sems_dim)

def resnet101(num_classes=1000, include_top=True, text_feature_dim=19, sems_dim=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, text_feature_dim=text_feature_dim, sems_dim=sems_dim)

def resnext50_32x4d(num_classes=1000, include_top=True, text_feature_dim=19, sems_dim=3):
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, groups=groups, width_per_group=width_per_group, text_feature_dim=text_feature_dim, sems_dim=sems_dim)

def resnext101_32x8d(num_classes=1000, include_top=True, text_feature_dim=19, sems_dim=3):
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, groups=groups, width_per_group=width_per_group, text_feature_dim=text_feature_dim, sems_dim=sems_dim)
