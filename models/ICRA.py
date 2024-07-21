import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('gen:\n:', net)
    print('generator Total number of parameters: %d' % num_params)



### GlobalGenerator for derain

class downsample_unit(nn.Module):
    def __init__(self, indim, outdim ):
        super(downsample_unit, self).__init__()
        downsample_list = [nn.Conv2d(indim, outdim, kernel_size=3, stride=2, padding=1),  #,3,2,1
            nn.BatchNorm2d(outdim), nn.ReLU(True)]
        self.model = nn.Sequential(*downsample_list)

    def forward(self, x):
        return self.model(x)


class upsample_unit(nn.Module):
    def __init__(self, indim, outdim ):
        super(upsample_unit, self).__init__()  
        # origin: 3,2,1,1  #output_padding=1
        upsample_list = [nn.ConvTranspose2d(indim, outdim, kernel_size=4, stride=2, padding=1), 
                    nn.BatchNorm2d(outdim), nn.ReLU(True)]
        self.model = nn.Sequential(*upsample_list)

        
    def forward(self, x):
        x = self.model(x)
        return x

class Derain_GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Derain_GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        init_unit = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.init_conv_unit = nn.Sequential(*init_unit)

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            setattr(self, 'down'+str(i), downsample_unit(ngf * mult,  ngf * mult*2))  #in  out 

        ### resnet blocks
        mult = 2**n_downsampling
        resblock_list = []
        for i in range(n_blocks):
            resblock_list += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.resblock_seq = nn.Sequential(*resblock_list)

        mult = 2**(n_downsampling - 0)
        setattr(self, 'up'+str(0), upsample_unit(ngf * mult,  int(ngf * mult / 2)) )
        mult = 2**(n_downsampling - 1)
        setattr(self, 'up'+str(1), upsample_unit(ngf * mult*2,  int(ngf * mult / 2)) )
        mult = 2**(n_downsampling - 2)
        setattr(self, 'up'+str(2), upsample_unit(ngf * mult*2,  int(ngf * mult / 2) ) )
        mult = 2**(n_downsampling - 3)
        setattr(self, 'up'+str(3), upsample_unit(ngf * mult,  int(ngf * mult / 2)) )

        output_list = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.out_unit = nn.Sequential(*output_list)
            
    def forward(self, input):
        x_init = self.init_conv_unit(input)
        down_unit0= getattr(self, 'down0'); d0= down_unit0(x_init)
        down_unit1= getattr(self, 'down1'); d1= down_unit1(d0)
        down_unit2= getattr(self, 'down2'); d2= down_unit2(d1)
        down_unit3= getattr(self, 'down3'); d3= down_unit3(d2)

        res =self.resblock_seq(d3)
        #print('resnet size:', res.shape, '\n')  # if input is 256*256 , then resnet output is 16*16

        up_uint0 = getattr(self, 'up0'); up0 = up_uint0(res)
        up0 = torch.cat((up0, d2), 1)
        up_uint1 = getattr(self, 'up1'); up1 = up_uint1(up0); 
        
        up1 =torch.cat((up1, d1), 1)
        up_uint2 = getattr(self, 'up2'); up2 = up_uint2(up1) ; 
        
        up_uint3 = getattr(self, 'up3'); up3 = up_uint3(up2)

        out = self.out_unit(up3)
        return out

        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




####create generator 16
def create_gen_nets():
    generator = Derain_GlobalGenerator(input_nc=3, output_nc=3, ngf=16, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect')
    return  generator

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

    net = create_gen_nets()
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