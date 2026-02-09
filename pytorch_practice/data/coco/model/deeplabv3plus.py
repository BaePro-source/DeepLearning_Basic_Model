import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabV3 핵심 모듈)"""
    def __init__(self, in_ch: int, out_ch: int = 256, rates=(6, 12, 18)):
        super().__init__()

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # 3x3 atrous conv branches
        self.branches_atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])

        # Image pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # projection
        total = out_ch * (2 + len(rates))  # 1x1 + atrous*len + image_pool
        self.project = nn.Sequential(
            nn.Conv2d(total, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 0.1 -> 0.5로 증가 (원본 논문 기준)
        )

    def forward(self, x):
        h, w = x.shape[2:]  # 더 명확한 표현

        feats = [self.branch1(x)]
        feats += [b(x) for b in self.branches_atrous]

        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(pool)

        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabv3(nn.Module):
    """
    DeepLabV3+ (semantic segmentation)
    - backbone: ResNet50 (ImageNet pretrained)
    - head: ASPP + decoder with low-level features
    - auxiliary loss head (optional)
    - output: [B, num_classes, H, W] or dict with aux logits
    """
    def __init__(
        self, 
        num_classes: int, 
        backbone: str = "resnet50", 
        pretrained: bool = True,
        use_aux_loss: bool = True,
        output_stride: int = 16
    ):
        super().__init__()

        if backbone != "resnet50":
            raise ValueError("현재 구현은 backbone='resnet50'만 지원합니다.")
        
        if output_stride not in [8, 16]:
            raise ValueError("output_stride는 8 또는 16이어야 합니다.")

        self.use_aux_loss = use_aux_loss

        # output_stride 설정
        if output_stride == 16:
            replace_stride = [False, False, True]
            aspp_rates = (6, 12, 18)
        else:  # output_stride == 8
            replace_stride = [False, True, True]
            aspp_rates = (12, 24, 36)

        # Backbone 구성
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        resnet = models.resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_stride,
        )

        # ResNet backbone에서 feature 뽑는 부분 구성
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # ASPP module
        self.aspp = ASPP(in_ch=2048, out_ch=256, rates=aspp_rates)

        # Low-level feature projection (from layer1)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        # Auxiliary classifier (from layer3, for training only)
        if use_aux_loss:
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1),
            )

    def forward(self, x):
        input_shape = x.shape[2:]  # (H, W)

        # Backbone forward
        x = self.stem(x)
        low_level_feat = self.layer1(x)  # 1/4 scale, 256 channels
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        aux_feat = x  # 1/8 or 1/16 scale, 1024 channels (for aux loss)
        x = self.layer4(x)  # 1/8 or 1/16 scale, 2048 channels

        # ASPP
        x = self.aspp(x)

        # Upsample ASPP output to match low-level features
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode="bilinear", align_corners=False)

        # Low-level feature projection
        low_level_feat = self.low_level_conv(low_level_feat)

        # Concatenate and decode
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)

        # Final upsampling to input resolution
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        # Auxiliary loss output (training only)
        if self.use_aux_loss and self.training:
            aux = self.aux_classifier(aux_feat)
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            return {"out": x, "aux": aux}
        
        return x


# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = DeepLabv3(num_classes=21, pretrained=True, use_aux_loss=True)
    model.eval()

    # 테스트 입력
    x = torch.randn(2, 3, 512, 512)
    
    # Inference mode
    with torch.no_grad():
        out = model(x)
        print(f"Output shape: {out.shape}")  # [2, 21, 512, 512]
    
    # Training mode (with aux loss)
    model.train()
    out = model(x)
    print(f"Main output shape: {out['out'].shape}")  # [2, 21, 512, 512]
    print(f"Aux output shape: {out['aux'].shape}")   # [2, 21, 512, 512]