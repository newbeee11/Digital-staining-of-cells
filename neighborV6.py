import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from utils import get_detail_channel

# ============================================================
# Utility
# ============================================================
def denorm_to_uint8(tensor):
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2 * 255
    return tensor.byte()

# ============================================================
# 基础模块 (统一为 GroupNorm + GELU)
# ============================================================



class ResidualConvBlock(nn.Module):
    def __init__(self, ch, use_se=False):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch,norm=False),
            ConvBlock(ch, ch,norm=False),
        )
        self.use_se = use_se
        self.se = SEBlock(ch) if use_se else None

    def forward(self, x):
        out = self.block(x)
        if self.use_se:
            out = self.se(out)
        return x + out

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, s=2)

    def forward(self, x):
        return self.block(x)

class FuseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 复用已有的反射填充模块
        self.main = nn.Sequential(
            ConvBlock(in_ch, out_ch, k=3, s=1, norm=False),
            ConvBlock(out_ch, out_ch, k=3, s=1, norm=False)
        )
        self.short = nn.Conv2d(in_ch, out_ch, 1, bias=False)  # 1x1 不需要 pad

    def forward(self, x):
        return self.main(x) + self.short(x)


def get_norm_layer(channels, groups=32):
    # 如果通道数不足以分32组，则回退到8组
    actual_groups = groups if channels % groups == 0 else 8
    return nn.GroupNorm(actual_groups, channels)


# ============================================================
# 基础模块：统一反射填充与抗锯齿
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, norm=True, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.block = nn.Sequential(
            nn.ReflectionPad2d(p),
            nn.Conv2d(in_ch, out_ch, k, s, padding=0, bias=False),
        )
        self.norm = get_norm_layer(out_ch) if norm else nn.Identity()
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.block(x)))


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):  # 减小还原率，增强通道敏感度
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.GELU(),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


# ============================================================
# 核心 Block 优化：增加平滑性
# ============================================================
class EnhancedEncoderBlock(nn.Module):
    def __init__(self, ch, expansion=4, use_se=True):
        super().__init__()
        # 1. 换回普通卷积 (Vanilla Convolution)
        # 使用两层 3x3 卷积构成的标准残差块，不再分离空间和通道
        mid_ch = ch * expansion

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, mid_ch, kernel_size=3, padding=0, bias=False),
            get_norm_layer(mid_ch),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(mid_ch, ch, kernel_size=1, padding=0, bias=False),
            get_norm_layer(ch)
        )

        # 2. 引入 LayerScale (这是消除初期伪影的神器)
        # 初始值设为 1e-6，让网络在训练初期几乎等效于 Identity 传输
        # 这能物理性地屏蔽掉卷积核权重未练好时产生的棋盘格或条纹
        self.layer_scale = nn.Parameter(1.0 * torch.ones(ch), requires_grad=True)

        # 3. 注意力机制 (由于你之前发现 SE 会导致条纹，默认设为 False)
        self.se = SEBlock(ch) if use_se else nn.Identity()

    def forward(self, x):
        resid = x

        # 主路径
        out = self.conv1(x)
        out = self.conv2(out)

        # 应用通道缩放
        out = out * self.layer_scale.view(1, -1, 1, 1)

        # 残差连接
        return resid + self.se(out)

class DualAttentionBottleneck(nn.Module):
    def __init__(self, ch, use_se=True):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)

        self.pad13 = nn.ReflectionPad2d((1, 1, 0, 0))
        self.dw13 = nn.Conv2d(ch, ch, (1, 3), padding=0, groups=ch, bias=False)
        self.pad31 = nn.ReflectionPad2d((0, 0, 1, 1))
        self.dw31 = nn.Conv2d(ch, ch, (3, 1), padding=0, groups=ch, bias=False)

        self.norm2 = get_norm_layer(ch)
        self.act2 = nn.GELU()
        self.se = SEBlock(ch) if use_se else nn.Identity()
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        feat = self.conv1(x)
        # local = self.dw31(self.pad31(self.dw13(self.pad13(feat))))
        # local = self.act2(self.norm2(local))
        # out = local + self.se(feat)
        return x + feat






# ============================================================
# 生成器：移除末端归一化，解决方框感
# ============================================================
class EnhancedResNetGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_bottleneck=3):
        super().__init__()

        # Encoder: 保持物理亮度的定量信息，第一层不使用 Norm
        self.enc1 = nn.Sequential(
            ConvBlock(input_nc, ngf, norm=False),
            ResidualConvBlock(ngf)
        )
        self.enc2 = nn.Sequential(DownBlock(ngf, ngf * 2), EnhancedEncoderBlock(ngf * 2))
        self.enc3 = nn.Sequential(DownBlock(ngf * 2, ngf * 4), EnhancedEncoderBlock(ngf * 4))
        self.enc4 = nn.Sequential(DownBlock(ngf * 4, ngf * 8), EnhancedEncoderBlock(ngf * 8))

        self.bottleneck = nn.Sequential(*[DualAttentionBottleneck(ngf * 8) for _ in range(n_bottleneck)])

        # Decoder: 关键优化
        # up1 还在深层，可以使用 Norm
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(ngf * 8, ngf * 4, norm=True)
        )
        # up2, up3 接近输出，必须移除 Norm 以消除由于单张 Patch 统计量导致的“方框感”
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(ngf * 4, ngf * 2, norm=False)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(ngf * 2, ngf, norm=False)
        )

        self.fuse_up1 = FuseBlock(ngf * 8, ngf * 4)
        self.fuse_up2 = FuseBlock(ngf * 4, ngf * 2)
        self.fuse_up3 = FuseBlock(ngf * 2, ngf)

        # 增加一个 1x1 卷积作为 Refine 层，用于微调最终颜色而不改动空间结构
        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, 3, 1, 0),
            nn.GELU(),
            nn.Conv2d(ngf, output_nc, 1),  # 1x1 卷积收尾
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_feats=False):

            # detail = detail.clamp(-1, 1)

        # 门控细节：只在有组织的区域增强细节
        energy = x.mean(dim=1, keepdim=True)
        xin=x
        x = torch.cat([x, energy], dim=1)

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        bottleneck = self.bottleneck(x4)

        out = self.up1(bottleneck)
        out = self.fuse_up1(torch.cat([out, x3], dim=1))
        out = self.up2(out)
        out = self.fuse_up2(torch.cat([out, x2], dim=1))
        out = self.up3(out)
        out = self.fuse_up3(torch.cat([out, x1], dim=1))
        bg = energy

        bg = bg - bg.mean(dim=(2, 3), keepdim=True)
        out=out+0.2*bg.detach()
        out_img = self.final(out)
        return (out_img, [x2, x3, x4, bottleneck]) if return_feats else out_img