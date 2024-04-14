from models.basenet import *
from numpy.testing import assert_array_almost_equal
import torch
import os
import random
import numpy as np
import sklearn.metrics as sk
from easydl import *

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

def get_model_mme(net, num_class=13, temp=0.05, top=False, norm=True):
    dim = 2048
    if "resnet" in net:
        model_g = ResBase(net, top=top)
        if "resnet18" in net:
            dim = 512
        if net == "resnet34":
            dim = 512
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
        dim = 4096
    if top:
        dim = 1000
    print("selected network %s"%net)
    return model_g, dim

class StochasticClassifier(nn.Module):

    def __init__(self, feature_dim, block_expansion, num_classes):
        super(StochasticClassifier, self).__init__()
        self.mean = nn.Linear(feature_dim * block_expansion, num_classes)
        self.std = nn.Linear(feature_dim * block_expansion, num_classes)
        
        # initialize weights
        self.apply(initialize_weights)

    def forward(self, x, mode='train'):
        if mode == 'test':
            weight = self.mean.weight
            bias = self.mean.bias
        else:            
            e_weight = torch.randn(self.mean.weight.data.size()).cuda()
            e_bias = torch.randn(self.mean.bias.data.size()).cuda()
            
            weight = self.mean.weight + self.std.weight * e_weight
            bias = self.mean.bias + self.std.bias * e_bias
        
        # print("x:", x.shape)
        # print("w", weight.shape)
        # print("b", bias.shape)
        out = torch.matmul(x, weight.t()) + bias
        return out

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y