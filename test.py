import os
import shutil
import sys
import argparse
import time
import itertools

import numpy as np
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.append('./')
from utils.util import set_prefix, write, add_prefix
from utils.FocalLoss import FocalLoss



def test(model, test_loader, criterion):
  model.eval()
  for data in test_loader:
    data = data.cuda()
    data = Variable(data, volatile=True)
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    print("\n5/5 test : {}\n".format(pred))

def load_dataset():
  testdir = os.path.join('./data/data_augu', 'test')
  mean = [0.5186, 0.5186, 0.5186]
  std = [0.1968, 0.1968, 0.1968]
  normalize = transforms.Normalize(mean, std)

  test_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
  ])

  test_dataset = ImageFolder(testdir, test_transforms)
  print('load data-augumentation dataset successfully!!!')
  test_loader = DataLoader(test_dataset,
                           batch_size=90,
                           shuffle=False,
                           num_workers=1,
                           pin_memory=True)

  return test_loader

criterion = nn.CrossEntropyLoss().cuda()
test_loader = load_dataset()
model = models.resnet101(pretrained=True)
model = DataParallel(model).cuda()

checkpoint = torch.load('./classifier/model_best_new.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

test(model, test_loader, criterion)
