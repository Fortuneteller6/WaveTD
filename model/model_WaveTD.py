# @Author: Fortuneteller
# @Date&Time: 2022-05-05 12:26:18

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse

def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src

# Cascade_Conv
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# DWT_Conv
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        yL, yH = self.wt(x)
        # 对高、宽保留输入尺寸的一半，防止出现尺寸不匹配
        # 解决方案参考：https://github.com/fbcotter/pytorch_wavelets/issues/24
        yL_new = yL[:, :, :int(h/2), :int(w/2)]
        y_HL = yH[0][:, :, 0, :int(h/2), :int(w/2)]
        y_LH = yH[0][:, :, 1, :int(h/2), :int(w/2)]
        y_HH = yH[0][:, :, 2, :int(h/2), :int(w/2)]
        x = torch.cat([yL_new, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x, yL, yH

# LayerNorm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# AFAA
class AFAA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, bias=False):
        super(AFAA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dw = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
    
        self.qkv = Down_wt(in_dim, in_dim * 3)

        self.project_out = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        temp_h = int(h / 2)
        temp_w = int(w / 2)

        x = self.dw(x)
        qkv, yL, yH = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        v1 = v

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=temp_h, w=temp_w)

        out = self.project_out(out + v1)

        return out, yL, yH

# DGFN
class DGFN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DGFN, self).__init__()

        self.dilagroup_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=2, dilation=2)
        )

        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)

        self.post_conv = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        x0 = self.dilagroup_conv(x)
        x1 = self.dwconv(x)
        x_ = F.gelu(x0) * x1
        x_ = self.post_conv(x_ + x)
        return x_

# WST
class WST(nn.Module):
    def __init__(self, dim, out_dim, num_heads, LayerNorm_type):
        super(WST, self).__init__()
        self.layernorm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AFAA(dim, dim, num_heads)
        self.layernorm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DGFN(dim, out_dim)

    def forward(self, x):
        x = self.layernorm1(x)
        x1, yL, yH = self.attn(x)
        x2 = self.layernorm2(x1)
        x2 = self.ffn(x2)

        return x2, yL, yH

# WSU
class WSU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WSU, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.inverseDWT = DWTInverse(mode='zero', wave='haar')
        self.group = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, yL, yH):
        x1 = self.up(x)
        x_idwt = self.group(self.inverseDWT((yL, yH)))
        x = self.outconv(torch.cat((x1, x_idwt), dim=1))
        return x + x1 + x_idwt

# FCN Head
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# WaveTD&WaveTD-Tiny
class WaveTD(nn.Module):
    def __init__(self, n_channels, n_classes, heads, deep_supervision=True, **kwargs):
        super(WaveTD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.heads = heads
        self.deep_supervision = deep_supervision

        # WaveTD
        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (WST(32, 64, heads[0], LayerNorm_type='WithBias'))
        self.down2 = (WST(64, 128, heads[1], LayerNorm_type='WithBias'))
        self.down3 = (WST(128, 256, heads[2], LayerNorm_type='WithBias'))
        self.down4 = (WST(256, 512, heads[3], LayerNorm_type='WithBias'))

        self.Up5 = WSU(512, 256)
        self.Up4 = WSU(256, 128)
        self.Up3 = WSU(128, 64)
        self.Up2 = WSU(64, 32)
        self.outc = (OutConv(32, n_classes))

        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # WaveTD-Tiny
        # self.inc = (DoubleConv(n_channels, 16))
        # self.down1 = (WST(16, 32, heads[0], LayerNorm_type='WithBias'))
        # self.down2 = (WST(32, 64, heads[1], LayerNorm_type='WithBias'))
        # self.down3 = (WST(64, 128, heads[2], LayerNorm_type='WithBias'))
        # self.down4 = (WST(128, 256, heads[3], LayerNorm_type='WithBias'))

        # self.Up5 = WSU(256, 128)
        # self.Up4 = WSU(128, 64)
        # self.Up3 = WSU(64, 32)
        # self.Up2 = WSU(32, 16)
        # self.outc = (OutConv(16, n_classes))

        # self.conv5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x1 = self.inc(x)
        x2, yL2, yH2 = self.down1(x1)
        x3, yL3, yH3 = self.down2(x2)
        x4, yL4, yH4 = self.down3(x3)
        x5, yL5, yH5 = self.down4(x4)

        d5 = self.Up5(x5, yL5, yH5)
        d4 = self.Up4(d5, yL4, yH4)
        d3 = self.Up3(d4, yL3, yH3)
        d2 = self.Up2(d3, yL2, yH2)
        logits = self.outc(d2)


        d_s1 = self.conv1(d2)
        d_s2 = self.conv2(d3)
        d_s2 = _upsample_like(d_s2, d_s1)
        d_s3 = self.conv3(d4)
        d_s3 = _upsample_like(d_s3, d_s1)
        d_s4 = self.conv4(d5)
        d_s4 = _upsample_like(d_s4, d_s1)
        d_s5 = self.conv5(x5)
        d_s5 = _upsample_like(d_s5, d_s1)
        if self.deep_supervision:
            outs = [d_s1, d_s2, d_s3, d_s4, d_s5, logits]
        else:
            outs = logits

        return outs


# # 示例用法
# from thop import profile
# model = WaveTD(n_channels=3, n_classes=1, heads=[1, 2, 4, 8])
# input = torch.randn(1, 3, 256, 256)
# Flops, Params = profile(model, inputs=(input, ))
# # 计算量
# print('Flops: % .4fG' % (Flops / 1000000000))
# # 参数量
# print('Params参数量: % .4fM' % (Params / 1000000))
