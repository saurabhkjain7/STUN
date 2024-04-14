from torchvision import models
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torch.nn as nn


class GradReverse(Function):
    lambd = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()*GradReverse.lambd


def grad_reverse(x, lambd=1.0):
    GradReverse.lambd = lambd
    return GradReverse.apply(x)

class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)

        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*mod)
      


    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), self.dim)
            return x


class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05):
        super(ResClassifier_MME, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, return_feat=False, reverse=False):
        if return_feat:
            return x
        if reverse==True:
            x = grad_reverse(x)
        x = F.normalize(x)
        x = self.fc(x)/self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)




class ResNet(nn.Module):
    def __init__(self, option='resnet50', pret=True, unit_size=100):
        super(ResNet, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.base = nn.Sequential(
            model_resnet.conv1,
            model_resnet.bn1,
            model_resnet.relu,
            model_resnet.maxpool,
            model_resnet.layer1,
            model_resnet.layer2,
            model_resnet.layer3,
            model_resnet.layer4,
            model_resnet.avgpool,
        )

    def forward(self, x, reverse=False, alpha=1):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        out = F.relu(x)
        return out
    

class VGGBase(nn.Module):
    def __init__(self, option='vgg', pret=True, no_pool=False, top=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        self.top = top

        if option =='vgg11_bn':
            vgg16=models.vgg11_bn(pretrained=pret)
        elif option == 'vgg11':
            vgg16 = models.vgg11(pretrained=pret)
        elif option == 'vgg13':
            vgg16 = models.vgg13(pretrained=pret)
        elif option == 'vgg13_bn':
            vgg16 = models.vgg13_bn(pretrained=pret)
        elif option == "vgg16":
            vgg16 = models.vgg16(pretrained=pret)
        elif option == "vgg16_bn":
            vgg16 = models.vgg16_bn(pretrained=pret)
        elif option == "vgg19":
            vgg16 = models.vgg19(pretrained=pret)
        elif option == "vgg19_bn":
            vgg16 = models.vgg19_bn(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        if self.top:
            self.vgg = vgg16

    def forward(self, x, source=True,target=False):
        if self.top:
            x = self.vgg(x)
            return x
        else:
            x = self.features(x)
            x = x.view(x.size(0), 7 * 7 * 512)
            x = self.classifier(x)
            return x
        
