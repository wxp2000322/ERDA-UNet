import torch
from torch import nn
from timm.models.layers import DropPath
import torch

from torch import nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from model.head import _FCNHead


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        queries = self.query_conv(x).view(batch_size, -1, height * width)
        keys = self.key_conv(x).view(batch_size, -1, height * width)
        values = self.value_conv(x).view(batch_size, -1, height * width)

        attention_weights = F.softmax(torch.bmm(queries.permute(0, 2, 1), keys), dim=-1)
        out = torch.bmm(values, attention_weights.permute(0, 2, 1)).view(batch_size, channels, height, width)

        return self.gamma * out


class EdgeAttention(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0.1):
        super(EdgeAttention, self).__init__()
        # 初始化Laplacian卷积核
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        # 创建可学习的卷积层
        self.laplacian_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.drop_path = DropPath(drop_path)
        # 将卷积核的初始权重设置为Laplacian算子
        with torch.no_grad():
            self.laplacian_conv.weight.copy_(laplacian_kernel)
            self.laplacian_conv.weight.requires_grad = True
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        # 使用深度可分离卷积来减少参数
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels,
                                        bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 添加卷积层以保持通道数不变
        self.conv_after_concat = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path)
        # 自注意力层
        self.self_attention = SelfAttention(out_channels)

    def forward(self, x):
        # 计算Laplacian边缘特征
        edge_features = self.laplacian_conv(x)  # 使用Laplacian卷积提取边缘
        # 使用深度可分离卷积提取主特征
        edge_features = self.depthwise_conv(edge_features)
        edge_features = self.pointwise_conv(edge_features)

        # 特征拼接
        combined_features = torch.cat((x, edge_features), dim=1)  # 按通道拼接

        # 添加卷积层以保持通道数不变
        combined_features = self.conv_after_concat(combined_features)
        # 添加归一化
        combined_features = self.batch_norm(combined_features)

        # 添加激活函数
        combined_features = self.activation(combined_features)
        # 应用自注意力
        attention_output = self.self_attention(combined_features)

        return attention_output + x# 残差连接
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features  # same as input
        hidden_features = hidden_features or in_features  # x4
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class DESABlock(torch.nn.Module):
    def __init__(self, dim, depth, shift_size, dropout=0.):
        super().__init__()
        self.backbone = nn.ModuleList([])
        for i in range(depth):
            self.backbone += [nn.Sequential(
                EdgeAttention(dim, dim),
                FFN(dim, dim * 4, drop_path=dropout),
            )]

    def forward(self, x):
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        return x

class EFBlock(nn.Module):
    def __init__(self, dim, depth, channel, patch_size, dropout=0.0):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv2 = nn.Conv2d(channel, dim, kernel_size=1, stride=1, padding=0, bias=False)


        self.transformer = DESABlock(dim, depth, dropout=dropout, shift_size=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU())

    def forward(self, x):
        y = x.clone()
        x = self.conv2(x)
        x = self.transformer(x)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

def autopad(kernel_size):
    return (kernel_size - 1) // 2


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, autopad(dw_size), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class Encoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2):
        super().__init__()
        self.ghost1 = GhostModule(inp, int(inp * 2), kernel_size)
        self.convdw = nn.Conv2d(in_channels=int(inp * 2),
                                out_channels=int(inp * 2),
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=autopad(kernel_size),
                                groups=int(inp * 2))
        self.bn = nn.BatchNorm2d(int(inp * 2))
        self.ghost2 = GhostModule(int(inp * 2), oup, kernel_size, stride=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride,
                      autopad(kernel_size), groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.convdw(x)
        x = self.bn(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden, oup, kernel_size=3):
        super().__init__()
        self.ghost = GhostModule(hidden, oup, kernel_size)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)  # 1,256,256
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.ghost(x1)
        return x1

class LWAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(EfficientAttention, self).__init__()
        self.reduction = reduction
        self.conv_query = nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 1. 通过卷积层生成查询、键和值
        query = self.conv_query(x).view(batch_size, -1, height * width)  # (batch_size, out_channels/reduction, height * width)
        key = self.conv_key(x).view(batch_size, -1, height * width)      # (batch_size, out_channels/reduction, height * width)
        value = self.conv_value(x).view(batch_size, -1, height * width)  # (batch_size, out_channels, height * width)

        # 2. 计算注意力权重
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)  # (batch_size, height * width, height * width)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 3. 计算加权值
        attention_output = torch.bmm(attention_weights, value.permute(0, 2, 1)).permute(0, 2, 1)  # (batch_size, height * width, out_channels)

        # 4. 通过卷积层进行输出变换
        attention_output = attention_output.view(batch_size, -1, height, width)  # (batch_size, out_channels, height, width)

        return attention_output
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU, downsample=False):
        super(CBR, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样层

    def forward(self, x):
        x = self.conv(x)  # 进行卷积操作
        if self.downsample:
            x = self.pool(x)  # 下采样
        return x
class ERDA(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer0 = Encoder(3, 128)

        self.layer1 = Encoder(128, 128)
        self.CBR1 = CBR(128, 64, downsample=True)
        self.AT1 = LWAttention(in_channels=64, out_channels=64)


        self.layer2 = Encoder(192, 192)
        self.CBR2 = CBR(192, 64, downsample=True)
        self.AT2 = LWAttention(in_channels=64, out_channels=64)

        self.EF = EFBlock(96, 6, 256, (2, 2), 0.1)

        self.decode2 = Decoder(256 + 192, 192)

        self.decode1 = Decoder(192 + 128, 128)

        self.head = _FCNHead(128, n_class)
        self.apply(self.__init_weights)
        self.vis = False

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)

    def forward(self, input):
        _, _, h, w = input.size()
        e0 = self.layer0(input)

        e1 = self.layer1(e0)
        c1 = self.CBR1(e0)
        a1 = self.AT1(c1)
        m1 = torch.cat((a1, e1), dim=1)

        e2 = self.layer2(m1)
        c2 = self.CBR2(m1)
        a2 = self.AT2(c2)
        m2 = torch.cat((a2, e2), dim=1)

        f = self.EF(m2)

        d2 = self.decode2(f, m1)
        d1 = self.decode1(d2, e0)
        out = F.interpolate(d1, size=[h, w], mode='bilinear', align_corners=True)
        out = self.head(out)
        return out if not self.vis else (e0, e1, e2, c1, c2,a1, a2,m1,m2,f,d2, d1, out)


if __name__ == "__main__":
    model = ERDA(1)
    from torchsummary import summary

    summary(model, (3, 256, 256), device='cpu')
