import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_detail_channel


# -------------------- AdaIN --------------------
class SpatialAdaIN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.eps = 1e-5

        self.style_scale = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, style):

        if x.shape[2:] != style.shape[2:]:
            style = F.interpolate(style, x.shape[2:], mode='bilinear', align_corners=False)

        mean = x.mean((2,3), keepdim=True)
        std = torch.sqrt(x.var((2,3), keepdim=True, unbiased=False)+self.eps)

        x_norm = (x - mean) / std

        gamma = self.style_scale(style)
        gamma = torch.tanh(gamma) * 0.5   # ⭐ 限幅

        modulated = x_norm * (1 + gamma)

        alpha = torch.sigmoid(self.alpha)

        return x + alpha * (modulated - x)
class StructureBuffer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # 1. 添加显式的反射填充层，padding=1 对应 3x3 卷积
        self.pad = nn.ReflectionPad2d(1)

        # 2. 将 Conv2d 内部的 padding 设为 0
        self.dw = nn.Conv2d(ch, ch, 3, 1, padding=0, groups=ch, bias=False)

        # 3. 1x1 卷积不需要 padding，保持不变
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # 先进行反射填充，再进行深度卷积
        out = self.dw(self.pad(x))
        out = self.pw(out)
        return x + self.act(out)

# -------------------- UNetFusionBlock --------------------
class UNetFusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()

        # 1. 先用 1x1 卷积把通道数降下来，同时强制混合 Deep 和 Skip 的信息
        # 这一步是关键！它像搅拌机一样把两种特征混合，
        # 使得后续层无法区分哪个是 Deep 哪个是 Skip，只能一起处理。
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),  # 必须加 Norm！
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1)
        )

        # 2. 然后再接一个标准的残差块或者卷积块来提取特征
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, 1, 0, bias=False),
            nn.GroupNorm(8, out_ch),  # 必须加 Norm！
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, 1, 0, bias=False),
            nn.GroupNorm(8, out_ch)  # 必须加 Norm！
        )

        # 3. 这里的 Shortcut 是针对混合后的特征的，而不是针对原始输入的
        # 这就是标准的 ResNet 结构了
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x: [Batch, in_ch, H, W] (Concat of Deep + Skip)

        # 步骤 1: 强制混合
        out = self.mix(x)

        # 步骤 2: 特征提取（带残差）
        res = self.conv(out)
        res = self.conv(out)
        out = self.act(out + 0.1 * res)  # 小系数
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=0, norm='in'):
        super().__init__()
        norm_layer = nn.GroupNorm(8, out_ch)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            norm_layer,
            nn. LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)
# -------------------- ResNetBlock --------------------
class ResNetBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),

            nn.Conv2d(ch, 2*ch, 3, bias=False),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.2),
            nn.ReflectionPad2d(1),
            nn.GroupNorm(32, 2*ch),
            nn.Conv2d(2*ch, ch, 3, bias=False),

            # nn.LeakyReLU(0.2, inplace=True),

        )
        # 加入 LayerScale 稳定初始梯度


    def forward(self, x):
        out=x +  0.5*self.conv(x)
        return  out

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # --- Channel Attention ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()


        self.spatial_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=0, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. 通道注意力分支
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        x = x * self.sigmoid_channel(avg + max_)

        # 2. 空间注意力分支 (修改点)
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp_input = torch.cat([avg_sp, max_sp], dim=1)

        # 先进行反射填充，再进行大核卷积
        sp_feat = self.conv_spatial(self.spatial_pad(sp_input))

        # 应用空间权重 (保持你原有的 1 + 0.3 * sigmoid 逻辑)
        x = x * (1 + 0.3 * self.sigmoid_spatial(sp_feat))
        return x


class ResidualUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 支路 A：复杂的纹理生成路径 (你原有的 PixelShuffle)
        self.main_path = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)
        )




    def forward(self, x):


        return  self.main_path(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_norm=True, use_cbam=False):
        super().__init__()
        # 1. 换成带残差的上采样
        self.upsample = ResidualUpsample(in_ch, out_ch)

        # 2. 上采样后的平滑与增强
        self.refine = nn.Sequential(
            nn.GroupNorm(8, out_ch) if use_norm else nn.Identity(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.1)
        )

        # 3. 之前漏掉的残差块 (建议保留，增强非线性)
        # self.residual_block = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
        #     nn.GroupNorm(8, out_ch) if use_norm else nn.Identity(),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
        # )
        # # self.res_gamma = nn.Parameter(0.1 * torch.ones(out_ch), requires_grad=True)

    def forward(self, x):
        # 上采样（带投影残差）
        x = self.upsample(x)
        x = self.refine(x)

        # 局部的深度残差增强
        # x = x +  self.residual_block(x)
        return x

class SpatialStyleEncoder(nn.Module):
    def __init__(self, input_nc=3, ngf=64):
        super().__init__()
        # 保持与 Center Encoder 同步的下采样频率
        # 目标是输出与 center_feat3 (Down2) 同样的分辨率 (H/4, W/4)
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.GroupNorm(8, ngf),  # <-- 修正：GroupNorm 移到 LeakyReLU 之前
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=0),  # H/2
            nn.GroupNorm(8, ngf * 2),  # <-- 第二次添加归一化
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=0),  # H/4
            nn.GroupNorm(8, ngf * 4), # <-- 第三次添加归一化
            nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, x):
        feat = self.model(x)  # [B, ngf*4, H/4, W/4]
        # 🟢 改进：把特征压缩到极小的分辨率，然后再恢复，以此过滤掉高频结构
        b, c, h, w = feat.shape
        coarse_feat = F.adaptive_avg_pool2d(feat, (45, 45))  # 压缩到 4x4
        smooth_feat = F.interpolate(coarse_feat, size=(h, w), mode='bilinear', align_corners=False)
        return smooth_feat


class BottleneckSkipAdapter(nn.Module):
    def __init__(
            self,
            in_ch,
            bottleneck_ratio=4,
            norm_groups=8,
            # 移除了固定的 scale=0.2
    ):
        super().__init__()
        mid_ch = in_ch // bottleneck_ratio

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(norm_groups, mid_ch), mid_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=False),
        )
        self.gate = nn.Parameter(0.01 * torch.ones(1, in_ch, 1, 1))
        # ★ 关键修改：初始化为 0 的可学习参数
        # 这样训练开始时，skip 输出为 0，梯度被迫走 Main Branch


    def forward(self, x):
        # 这里的 x 是来自 Encoder 的特征
        return 0.2*self.proj(x)+self.gate*self.proj(x)

class NucleusRes(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, 1, bias=False)
        self.norm = nn.InstanceNorm2d(C, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))
def remove_intensity(x, eps=1e-6):
    # x: [-1,1]
    intensity = x.abs().mean(dim=1, keepdim=True)
    x_norm = x / (intensity + eps)
    return x_norm.clamp(-1, 1), intensity
def highpass(x):
    return x - F.avg_pool2d(x, 5, 1, 2)


class CenterDownsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, bias=False),
            # 这里的 Norm 层是 block[3]
            nn.GroupNorm(min(32, out_ch // 4), out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        # 1. 卷积层初始化
        nn.init.kaiming_normal_(self.block[2].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        # 2. ★关键修复★：强制初始化 GroupNorm
        # GroupNorm 的权重在 block[3]
        nn.init.ones_(self.block[3].weight)  # Gamma = 1
        nn.init.zeros_(self.block[3].bias)  # Beta = 0

    def forward(self, x):
        return self.block(x)

def keep_variance(x, eps=0.05):
    std = x.std(dim=(2, 3), keepdim=True)

    correction = torch.relu(eps - std).detach()

    return x + correction * x
# -------------------- DualResNetGenerator (编码器减少归一化) --------------------
class DualResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=3, alpha=0.7):
        super().__init__()
        self.alpha = alpha

        # -------- Center Encoder --------
        self.center_conv1_rgb = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.center_conv1_detail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, ngf, 7),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.center_down1 = CenterDownsample(ngf, ngf * 2)
        self.center_down2 = CenterDownsample(ngf * 2, ngf * 4)
        self.center_down3 = CenterDownsample(ngf * 4, ngf * 8)

        self.style_encoder = SpatialStyleEncoder(input_nc, ngf=ngf)

        # ---------- Bottleneck ----------
        self.adain = SpatialAdaIN(ngf * 4)
        self.resblocks = nn.Sequential(*[ResNetBlock(ngf * 8) for _ in range(n_blocks)])

        # ---------- Decoder ----------
        self.up1 = DecoderBlock(ngf * 8, ngf * 4, use_norm=True)
        self.up2 = DecoderBlock(ngf * 4, ngf * 2, use_norm=False)
        self.up3 = DecoderBlock(ngf * 2, ngf, use_norm=False)

        # ---------- ★ 关键修改：像素级分类输出头 ★ ----------

        self.final_nucleus = nn.Sequential(
            nn.ReflectionPad2d(2),
            # 🔴 核心修改：ngf 改为 ngf + 1
            nn.Conv2d(ngf , ngf // 2, kernel_size=5, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf // 2, output_nc, kernel_size=3, padding=0),
            nn.Tanh()
        )


        self.detailfusion = UNetFusionBlock(ngf * 2, ngf, use_norm=False)

        # -------- Skip Fusion ----------
        self.fuse_up0 = UNetFusionBlock(ngf * 16, ngf * 8, use_norm=False)
        self.fuse_up1 = UNetFusionBlock(ngf * 8, ngf * 4, use_norm=False)
        self.fuse_up2 = UNetFusionBlock(ngf * 4, ngf * 2, use_norm=False)
        self.fuse_up3 = UNetFusionBlock(ngf * 2, ngf, use_norm=False)
        self.skip4_adapter = BottleneckSkipAdapter(
            in_ch=ngf * 8,
            bottleneck_ratio=4,

        )
        self.skip3_adapter = BottleneckSkipAdapter(
            in_ch=ngf * 4,
            bottleneck_ratio=4,

        )
        self.skip2_adapter = BottleneckSkipAdapter(
            in_ch=ngf * 2,
            bottleneck_ratio=4,

        )
        self.skip1_adapter = BottleneckSkipAdapter(
            in_ch=ngf ,
            bottleneck_ratio=4,

        )
        # ---------- Nucleus existence head ----------
        self.nucleus_blob_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf* 4, ngf , 3, bias=False),
            nn.InstanceNorm2d(ngf , affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf , ngf , 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.Conv2d(ngf , 1, 1,padding=0, bias=False)
        )
        self.nucleus_edge_head = NucleusRes(ngf*4)   # (B,1,H,W)




        self.refine = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, 3, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, 3, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )


    def apply_skip_dropout(self, x, drop_prob=0.5):
        """
        ★ 核心功能：随机丢弃跳跃连接特征。
        - 训练时：50% 概率全零（强制走 bottleneck），50% 概率乘以2（补偿数值）。
        - 测试时：原样输出。
        """
        if self.training and drop_prob > 0:
            keep_prob = 1.0 - drop_prob
            # 生成 Batch 级别的掩码，形状 (B, 1, 1, 1)
            mask = torch.bernoulli(torch.full((x.shape[0], 1, 1, 1), keep_prob, device=x.device))
            # 乘以 1/keep_prob 保持数学期望一致
            return x * mask / keep_prob
        return x
    def forward(self, neighbor, center, return_feats=False):
        xin = center
        center_norm, center_intensity = remove_intensity(center)

        energy = center_intensity
        # energy = center.abs().mean(dim=1, keepdim=True)

        feats = []
        s_feat = self.style_encoder(neighbor)

        # ---------------- Center encoder ----------------
        center_feat1 = self.detailfusion(
            torch.cat([self.center_conv1_rgb(center_norm), self.center_conv1_detail(energy)], dim=1)
        )
        fused_feat1 = center_feat1

        center_feat2 = self.center_down1(fused_feat1)
        fused_feat2 = center_feat2
        if return_feats: feats.append(fused_feat2)

        center_feat3 = self.center_down2(fused_feat2)
        # 1️⃣ Blob head 输出
        raw_blob = self.nucleus_blob_head(center_feat3)


        blob = raw_blob

        # 边界 = Laplacian / 高通
        edge = blob - F.avg_pool2d(blob, 3, 1, 1)

        # 分离两个 gate
        blob_gate = torch.sigmoid(2.0 * blob)


        # 融合（关键）
        nucleus_gate = blob_gate.clamp(0.05, 0.95)



        if return_feats: feats.append(center_feat3)

        center_feat4 = self.center_down3(center_feat3)
        fused_feat4 = center_feat4
        if return_feats: feats.append(fused_feat4)

        # ---------------- Bottleneck ----------------
        x = self.resblocks(fused_feat4)

        x = keep_variance(x)
        if return_feats: feats.append(x)
        # skip_feat4 = self.apply_skip_dropout(center_feat4, drop_prob=0.5)
        # x = torch.cat([x, skip_feat4], dim=1)
        # x = self.fuse_up0(x)
        # --------- Decoder + Skip Fusion ---------
        x = self.up1(x)
        p = 0.4*nucleus_gate  # (B,1,H,W)

        # res = self.nucleus_edge_head(x)  # (B,C,H,W)

        x = x +  p * ( x)
        skip_feat3 = self.apply_skip_dropout(center_feat3, drop_prob=0.0)
        skip_feat3 = self.skip3_adapter(skip_feat3)
        x = torch.cat([x, skip_feat3], dim=1)
        x = self.fuse_up1(x)
        x = self.adain(x, s_feat)

        # Layer 2
        x = self.up2(x)  # -> ngf*2
        # 浅层特征丢弃概率可以稍低，或者保持 0.5
        skip_feat2 = self.apply_skip_dropout(center_feat2, drop_prob=0.0)
        skip_feat2= self.skip2_adapter(skip_feat2)
        x = torch.cat([x, skip_feat2], dim=1)
        x = self.fuse_up2(x)

        # Layer 3
        x = self.up3(x)  # -> ngf
        skip_feat1 = self.apply_skip_dropout(center_feat1, drop_prob=0.)
        skip_feat1 = self.skip1_adapter(skip_feat1)
        x = torch.cat([x, skip_feat1], dim=1)
        x = self.fuse_up3(x)


        nucleus_prob_2 = F.interpolate(nucleus_gate, size=x.shape[2:], mode="bilinear", align_corners=False)

        nucleus_prob_inter = nucleus_prob_2


        x = self.refine(x)
        nucleus_prob = nucleus_prob_inter.clamp(0.05, 0.95)  # 也可以继续使用最后的核概率

        # local_mean = F.avg_pool2d(energy, kernel_size=11, stride=1, padding=5)
        # bg_detail = energy - local_mean  # 此时数值已经在 0 附近波动
        #
        # # 2. 放大波动，并用 tanh 限幅（真正发挥 tanh 的作用）
        # # 乘 3.0 是为了让微弱的纹理被放大，并通过 tanh 压制过亮的噪点
        # bg_norm = torch.tanh(3.0 * bg_detail)
        #
        # x = torch.cat([x, bg_norm], dim=1)


        # 最终输出
        out = self.final_nucleus(x)


        return (out, feats, nucleus_prob) if return_feats else (out, nucleus_prob)

