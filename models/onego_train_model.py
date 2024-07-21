import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
from PIL import Image
from copy import deepcopy
from torchvision import models
from torch.autograd import Variable
from models.onego_se_nets import SEBasicBlock, InsNorm
from models.onego_ops_derain import *
from collections import namedtuple
from models.onego_genotypes_searched import architectures

Genotype = namedtuple('Genotype', 'normal normal_concat')


class AugmentCell(nn.Module):

    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction, expansion_p, expansion):
        super(AugmentCell, self).__init__()


        if reduction_p   : self.preprocess0 = FactorizedReduce(C_pp, C, 2)
        elif expansion_p : self.preprocess0 = FactorizedExpand(C_pp, C, 1)
        else             : self.preprocess0 = FactorizedReduce(C_pp, C, 1)
        self.preprocess1 = StdConv(C_p, C, 1, 1, 0)

        if reduction:
            op_names, indices, values = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        elif expansion:
            op_names, indices, values = zip(*genotype.expand)
            concat = genotype.expand_concat
        else:
            op_names, indices, values = zip(*genotype.normal)
            concat = genotype.normal_concat
        self.expansion = len(concat)
        self._compile(C, op_names, indices, values, concat, reduction, expansion)
        self.configs = {'genotype': deepcopy(genotype),
                        'C_pp'    : deepcopy(C_pp),
                        'C_p'     : deepcopy(C_p),
                        'C'       : deepcopy(C),
                        'reduction_p': deepcopy(reduction_p),
                        'reduction': deepcopy(reduction),
                        'expansion_p': deepcopy(expansion_p),
                        'expansion': deepcopy(expansion)}

    def _compile(self, C, op_names, indices, values, concat, reduction, expansion):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            expand = True if expansion and index < 2 else False
            op = OPS[name](C, stride, True, expand)
            self._ops += [op]
        self._indices = indices
        self._values = values

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


                

class MFAttention_concat(nn.Module):
    def __init__(self, channels, r):
        super(MFAttention_concat, self).__init__()
        inter_channels = int(channels // r)

        self.local_conv1 = nn.Conv2d(channels, inter_channels, kernel_size=5, stride=1,padding=2)
        self.local_bn1   = nn.BatchNorm2d(inter_channels)
        self.local_relu  = nn.ReLU()
        self.local_conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=5, stride=1,padding=2)
        self.local_bn2   = nn.BatchNorm2d(inter_channels)
        self.local_conv3 = nn.Conv2d(inter_channels, inter_channels, kernel_size=5, stride=1,padding=2)
        self.local_bn3   = nn.BatchNorm2d(inter_channels)
        self.local_conv4 = nn.Conv2d(inter_channels, 3, kernel_size=5, stride=1,padding=2)
        self.local_bn4   = nn.BatchNorm2d(3)
        self.tanh = nn.Tanh()

    def forward(self, upfeat1, upfeat2, lofeat1, lofeat2):
        xa = torch.cat((upfeat1, upfeat2, lofeat1, lofeat2), dim=1)
        xl = self.local_conv1(xa)
        xl = self.local_bn1(xl)
        xl = self.local_relu(xl)
        xl = self.local_conv2(xl)
        xl = self.local_bn2(xl)
        xl = self.local_relu(xl)
        xl = self.local_conv3(xl)
        xl = self.local_bn3(xl)
        xl = self.local_relu(xl)
 
        xl = self.local_conv4(xl)
        xl = self.local_bn4(xl)
        out = self.tanh(xl)

        return out


class Raincleaner_train(nn.Module):
    def __init__(self, genotype, test_with_multigpus=False):
        super(Raincleaner_train, self).__init__()

        self.genotype = genotype
        # Initial convolutional layers
        self.conv1 = ConvLayer(3, 64, kernel_size=3, stride=1)
        self.norm1 = FeatNorm("batch_norm", 64)
        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.norm2 = FeatNorm("batch_norm", 64)
        self.mult_gpu_test = test_with_multigpus
        layer_reductions = [False] + [False] + [False]
        layer_expansions = [False] + [False] + [False]
        cells = []

        C_pp, C_p, C_curr = 64, 64, 64
        reduction_p, expansion_p = False, False
        for index, (reduction, expansion) in enumerate( zip(layer_reductions, layer_expansions)):
            cell = AugmentCell(genotype, C_pp, C_p, C_curr, reduction_p, reduction, expansion_p, expansion)
            reduction_p = reduction
            expansion_p = expansion
            cells.append(cell)
            C_pp, C_p = C_p, cell.multiplier*C_curr

        self.cells          = nn.ModuleList(cells)
        self.set_drop_path_prob(0)
        self.conv3 = ConvLayer(256, 64, kernel_size=3, stride=1)
        self.norm3 = FeatNorm("batch_norm", 64)
        self.conv4 = ConvLayer(64, 3, kernel_size=3, stride=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def set_drop_path_prob(self, value):
        self.drop_path_prob = value

    def get_weights(self):
        lists = list(self.conv1.parameters()) + list(self.norm1.parameters()) + list(self.conv2.parameters()) + list(self.norm2.parameters()) + list(self.cells.parameters()) + list(self.conv3.parameters()) + list(self.norm3.parameters()) + list(self.conv4.parameters())
        return lists

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))

        s0 = s1 = out
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)


        out = self.relu(self.norm3(self.conv3(s1)))
        out = self.tanh(self.conv4(out))
        if self.mult_gpu_test:
            out = out.cuda(0)
        out = out + x

        return out


       
class Raincleaner_wosha(nn.Module):
    def __init__(self, rs_genotype, rd_genotype):
        super(Raincleaner_wosha, self).__init__()

        self.rs_genotype = rs_genotype
        self.rd_genotype = rd_genotype

        # Initial convolutional layers
        self.rs_block_1 = SubModule(rs_genotype)
        self.rs_block_2 = SubModule(rs_genotype)
        self.rd_block_1 = SubModule(rd_genotype)
        self.rd_block_2 = SubModule(rd_genotype)

        self.fusion = MFAttention_concat(channels=256, r=4)
        self.relu = nn.ReLU()


    def get_weights(self):
        lists = self.rs_block_1.get_weights() + self.rd_block_1.get_weights() + self.rs_block_2.get_weights() + self.rd_block_2.get_weights() + list(self.fusion.parameters())
        return lists

    def forward(self, x):

        rs_residual_1, rs_feats_1 = self.rs_block_1(x)
        rs_out_1 = rs_residual_1 + x

        rd_residual_1, rd_feats_1 = self.rd_block_1(rs_out_1)
        rd_out_1 = rd_residual_1 + rs_out_1

        rd_residual_2, rd_feats_2 = self.rd_block_2(x)
        rd_out_2 = rd_residual_2 + x

        rs_residual_2, rs_feats_2 = self.rs_block_2(rd_out_2)
        rs_out_2 = rs_residual_2 + rd_out_2

        fused_rds = self.fusion(rs_feats_1,rd_feats_1,rd_feats_2,rs_feats_2)
        final_out = x + fused_rds

        return rs_out_1, rs_out_2, rd_out_1, rd_out_2, final_out

class Raincleaner_share(nn.Module):
    def __init__(self, rs_genotype, rd_genotype):
        super(Raincleaner_share, self).__init__()

        self.rs_genotype = rs_genotype
        self.rd_genotype = rd_genotype

        # Initial convolutional layers
        self.rs_block_1 = SubModule(rs_genotype)
        self.rd_block_1 = SubModule(rd_genotype)

        self.fusion = MFAttention_concat(channels=256, r=4)
        self.relu = nn.ReLU()


    def get_weights(self):
        lists = self.rs_block_1.get_weights() + self.rd_block_1.get_weights() + list(self.fusion.parameters())
        return lists

    def forward(self, x):

        rs_residual_1, rs_feats_1 = self.rs_block_1(x)
        rs_out_1 = rs_residual_1 + x

        rd_residual_1, rd_feats_1 = self.rd_block_1(rs_out_1)
        rd_out_1 = rd_residual_1 + rs_out_1

        rd_residual_2, rd_feats_2 = self.rd_block_1(x)
        rd_out_2 = rd_residual_2 + x

        rs_residual_2, rs_feats_2 = self.rs_block_1(rd_out_2)
        rs_out_2 = rs_residual_2 + rd_out_2

        fused_rds = self.fusion(rs_feats_1,rd_feats_1,rd_feats_2,rs_feats_2)
        final_out = x + fused_rds

        return rs_out_1, rs_out_2, rd_out_1, rd_out_2, final_out




class SubModule(nn.Module):
    def __init__(self, genotype, test_with_multigpus=False):
        super(SubModule, self).__init__()
        
        self.genotype = genotype
        # Initial convolutional layers
        self.conv1 = ConvLayer(3, 64, kernel_size=3, stride=1)
        self.norm1 = FeatNorm("batch_norm", 64)
        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.norm2 = FeatNorm("batch_norm", 64)
        self.mult_gpu_test = test_with_multigpus
        layer_reductions = [False] + [False] #+ [False] + [False] + [False]
        layer_expansions = [False] + [False] #+ [False] + [False] + [False]
        cells = []

        C_pp, C_p, C_curr = 64, 64, 64
        reduction_p, expansion_p = False, False
        for index, (reduction, expansion) in enumerate( zip(layer_reductions, layer_expansions)):
            cell = AugmentCell(genotype, C_pp, C_p, C_curr, reduction_p, reduction, expansion_p, expansion)
            reduction_p = reduction
            expansion_p = expansion
            cells.append(cell)
            C_pp, C_p = C_p, cell.multiplier*C_curr

        self.cells          = nn.ModuleList(cells)
        self.set_drop_path_prob(0)

        self.conv3 = ConvLayer(256, 64, kernel_size=3, stride=1)
        self.norm3 = FeatNorm("batch_norm", 64)        
        self.conv4 = ConvLayer(64, 3, kernel_size=3, stride=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def set_drop_path_prob(self, value):
        self.drop_path_prob = value

    def get_weights(self):
        lists = list(self.conv1.parameters()) + list(self.norm1.parameters()) + list(self.conv2.parameters()) + list(self.norm2.parameters()) + list(self.cells.parameters()) + list(self.conv3.parameters()) + list(self.norm3.parameters()) + list(self.conv4.parameters())
        return lists

    def forward(self, x):        
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))  
    
        s0 = s1 = out
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)


        feat_out = self.norm3(self.conv3(s1))
        out = self.relu(feat_out)                
        out = self.tanh(self.conv4(out))

        return out, feat_out   



        
#---------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)        
        return out
        
        
class FeatNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super(FeatNorm, self).__init__()
        if norm_type == "instance":
            self.norm = InsNorm(dim)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise Exception("Normalization type incorrect.")

    def forward(self, x):
        out = self.norm(x)        
        return out
def params_count(net):
    list1 = []
    for p in net.parameters():
        # print('p-',p.shape)
        list1.append(p)
    # print('len(net.parameters)',len(list1))
    n_parameters = sum(p.numel() for p in net.parameters())
    print('-----Model param: {:.5f}M'.format(n_parameters / 1e6))
    # print('-----Model memory: {:.5f}M'.format(n_parameters/1e6))
    return n_parameters
if __name__ == "__main__":
    resolution = 64
    # latent_size = resolution//8
    # net = My_DiT_test(input_size=resolution)
    genotype = architectures['RD_V2']
    net = Raincleaner_train(genotype)
    print(params_count(net))
    # print(net)
    with torch.no_grad():
        
        # x = torch.ones(4 * 6 * 3 * resolution * resolution).reshape(4, 6, 3, resolution, resolution)
        x = torch.ones(4 * 3 * resolution * resolution).reshape(4, 3, resolution, resolution)
        # y = torch.ones(4)
        # a = Variable(a.cuda)
        print('input=', x.shape)
        # a,b = net(x)
        # print('a,b=',a.shape,b.shape)
        a = net(x)
        print('output=', a.shape)