import os
import torch
from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat')


TEST_V9= Genotype(normal=[('dil_conv_11x5', 1, 0.2419893443584442), ('sep_conv_11x5', 0, 0.050061292946338654), ('sep_conv_7x5', 2, 0.2651098370552063), ('sep_conv_11x7', 1, 0.19399669766426086), ('sep_conv_5x3', 2, 0.2041405737400055), ('sep_conv_11x7', 1, 0.17715853452682495), ('sep_conv_7x5', 2, 0.16644716262817383), ('sep_conv_11x7', 3, 0.1611025631427765)], normal_concat=range(2, 6))

TRAIN_V1 = Genotype(normal=[('skip_connect', 1, 0.8828375935554504), ('SpatialAttention', 0, 0.16674959659576416), ('SpatialAttention', 0, 0.8944082260131836), ('dil_conv_7x5', 2, 0.6518889665603638), ('sep_conv_5x3', 3, 0.9358363747596741), ('sep_conv_5x3', 2, 0.93081134557724), ('sep_conv_5x3', 4, 0.9938530921936035), ('sep_conv_5x3', 3, 0.8889201879501343)], normal_concat=range(2, 6))

RD_V2 = Genotype(normal=[('sep_conv_7x5', 1, 0.6270664930343628), ('SpatialAttention', 0, 0.5707323551177979), ('SpatialAttention', 0, 0.6723378896713257), ('sep_conv_5x3', 1, 0.5788437128067017), ('sep_conv_5x3', 3, 0.5908498764038086), ('sep_conv_7x5', 1, 0.5440950393676758), ('sep_conv_5x3', 4, 0.5466906428337097), ('sep_conv_5x3', 3, 0.5305180549621582)], normal_concat=range(2, 6))

architectures = \
    {'TEST_V9'     : TEST_V9,
     'RD_V2'       :RD_V2,
     'TRAIN_V1'    :TRAIN_V1}