import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50

class Model(nn.Module):
    def __init__(self, last_conv_stride=2, local_conv_out_channels=128, identity_classes=None):
        super(Model, self).__init__()
        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

        self.local_feat_extractor = nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_classes = identity_classes
        if identity_classes is not None:
            self.fc = nn.Linear(2048, identity_classes)

    def forward(self, x):
        # shape [N, C, H, W]
        feat = self.base(x)
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        global_feat = global_feat.view(x.size(0), -1)

        # shape [N, C, H, 1]
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_feat_extractor(local_feat)

        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

        if self.identity_classes is not None:
            logits = self.fc(global_feat)
            return global_feat, local_feat, logits

        return global_feat, local_feat