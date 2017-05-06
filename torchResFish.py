import torch
import torchvision
from torch.autograd import Variable

from tqdm import *

resnet = torchvision.models.resnet18(pretrained=True)
