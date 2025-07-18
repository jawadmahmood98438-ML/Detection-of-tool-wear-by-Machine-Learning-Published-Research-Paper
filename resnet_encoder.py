from torchvision import models
import torch.nn as nn

def get_resnet18_encoder():
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]
    encoder = nn.Sequential(*modules)
    return encoder