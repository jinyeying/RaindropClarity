import torch
import torch.nn as nn
from copy import deepcopy as copy



Search_Space = {
    'base': ['none', 'skip_connect', 'sep_conv_5x3', 'sep_conv_7x5', 'sep_conv_11x5', 'sep_conv_11x7', 'dil_conv_7x5', 'dil_conv_11x5', 'dil_conv_11x7', 'ChannelAttention', 'SpatialAttention']
    }

OPS = {
  'none': lambda C, stride, affine, expand: Zero(stride, expand),
  'skip_connect': lambda C, stride, affine, expand: \
    FactorizedExpand(C, C, stride, affine=affine) if expand==True and stride == 1 else Identity() if expand==False and stride == 1 else FactorizedReduce(C, C, stride, affine=affine),
  'sep_conv_5x3': lambda C, stride, affine, expand: ConvLs(C, C, 5, 3, stride, 1, affine=affine, expand=expand),
  'sep_conv_7x5': lambda C, stride, affine, expand: ConvLs(C, C, 7, 5, stride, 1, affine=affine, expand=expand),
  'sep_conv_11x5': lambda C, stride, affine, expand: ConvLs(C, C, 11, 5, stride, 1, affine=affine, expand=expand),
  'sep_conv_11x7': lambda C, stride, affine, expand: ConvLs(C, C, 11, 7, stride, 1, affine=affine, expand=expand),
  'dil_conv_7x5': lambda C, stride, affine, expand: ConvLs(C, C, 7, 5, stride, 2, affine=affine, expand=expand), 
  'dil_conv_11x5': lambda C, stride, affine, expand: ConvLs(C, C, 11, 5, stride, 2, affine=affine, expand=expand), 
  'dil_conv_11x7': lambda C, stride, affine, expand: ConvLs(C, C, 11, 7, stride, 2, affine=affine, expand=expand),
  'ChannelAttention': lambda C, stride, affine, expand: ChannelAttention(C, 16),
  'SpatialAttention': lambda C, stride, affine, expand: SpatialAttention(7)
}

class ChannelAttention(nn.Module):
    def __init__(self, C, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x + x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.sigmoid(x)
        out = out * x + x
        return out


class ConvLs(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size_1, kernel_size_2, stride, dilation, affine, expand):
        super(ConvLs, self).__init__()
        reflect_padding = int(dilation * (kernel_size_1 - 1) / 2)
        self.reflection_pad_1 = nn.ReflectionPad2d(reflect_padding)
        self.conv2d_1 = nn.Conv2d(in_dim, out_dim, kernel_size_1, stride, dilation=dilation)
        self.bn_1     = nn.BatchNorm2d(out_dim, affine=affine)

        padding_2             = int((kernel_size_2 - 1) / 2)
        self.reflection_pad_2 = nn.ReflectionPad2d(padding_2)
        self.conv2d_2 = nn.Conv2d(in_dim, out_dim, kernel_size_2, stride)
        self.bn_2     = nn.BatchNorm2d(out_dim, affine=affine)

        self.relu   = nn.ReLU()

    def forward(self, x):
        x_r = x
        out = self.reflection_pad_1(x)
        out = self.conv2d_1(out)
        out = self.bn_1(out)
        out = out + x_r
        out = self.relu(out)

        out_2 = self.reflection_pad_2(out)
        out_2 = self.conv2d_2(out_2)
        out_2 = self.bn_2(out_2)
        out_2 = out_2 + x_r
        out_2 = self.relu(out_2)
   
        return out_2


class Identity(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):
  def __init__(self, stride, expand=False):
    super().__init__()
    self.expand = expand
    if self.expand:
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.stride = stride

  def forward(self, x):
    if self.stride == 1 and not self.expand:
        return x * 0.
    if self.expand and self.stride ==1:
        x = self.up(x)
        return x * 0.

    # re-sizing by stride
    return x[:, :, ::self.stride, ::self.stride] * 0.    


class FactorizedExpand(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, stride=1, affine=True):
        super().__init__()
        self.stride = stride
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu   = nn.ReLU()
        self.conv1  = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
        self.conv2  = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
        self.bn     = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        out = self.bn(out)
        return out


class FactorizedReduce(nn.Module):
  """
  Reduce feature map size by factorized pointwise(stride=2).
  """
  def __init__(self, C_in, C_out, stride=2, affine=True):
    super().__init__()
    self.stride = stride
    self.relu   = nn.ReLU()
    self.conv1  = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
    self.conv2  = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding=0, bias=False)
    self.bn     = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    if self.stride == 2:
      out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
    else:
      out = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
    out = self.bn(out)
    return out

class StdConv(nn.Module):
  """ Standard conv
  ReLU - Conv - BN
  """
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super().__init__()
    self.net = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.net(x)


class MixedOp(nn.Module):
  """ Mixed operation """
  def __init__(self, C, stride, expand, space_name):
    super().__init__()
    self.PRIMITIVES = copy(Search_Space[space_name])
    self._ops = nn.ModuleList()
    for primitive in self.PRIMITIVES:
      op = OPS[primitive](C, stride, affine=False, expand=expand)
      self._ops.append(op)

  def forward(self, x, weights):
    """
    Args:
      x: input
      weights: weight for each operation
    """
    return sum(w * op(x) for w, op in zip(weights, self._ops))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x