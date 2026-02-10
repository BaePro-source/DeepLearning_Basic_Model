"""
DeepLabV3+ Model Definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_ch: int, out_ch: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branches_atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        total = out_ch * (2 + len(rates))
        self.project = nn.Sequential(
            nn.Conv2d(total, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.branch1(x)]
        feats += [b(x) for b in self.branches_atrous]
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(pool)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet50 backbone"""
    def __init__(self, num_classes: int, pretrained: bool = True, output_stride: int = 16):
        super().__init__()
        if output_stride == 16:
            replace_stride = [False, False, True]
            aspp_rates = (6, 12, 18)
        elif output_stride == 8:
            replace_stride = [False, True, True]
            aspp_rates = (12, 24, 36)
        else:
            raise ValueError("output_stride must be 8 or 16")

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights, replace_stride_with_dilation=replace_stride)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(2048, 256, rates=aspp_rates)

        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_hw = x.shape[2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=False)

        low = self.low_proj(low)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
        return x