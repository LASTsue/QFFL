
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CNN import CNN
from mni_QFNN import Qfnn
# from tri_QFNN import Q_tnorm
# from ti_QFNN import Q_tnorm
from utils import get_logger,setup_seed,draw_loss,draw_acc,acc_cal
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from tqdm import tqdm

DEVICE=torch.device('cuda')
setup_seed(777)

transform = transforms.Compose([transforms.ToTensor()])

data_train = datasets.MNIST(root = "data/mnist/train",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="data/mnist/test",
                           transform = transform,
                           train = False,
                           download = True)


data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 5000,
                                                shuffle = True,
                                                 num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                            batch_size = 5000,
                                               shuffle = True,
                                                num_workers=2)

CNN_model=torch.load('result/model/CNN.pkl')
CNN_model=CNN_model.to(DEVICE)
CNN_model.eval()

train_data=[]
train_label=[]
for its,(x,y) in enumerate(tqdm(data_loader_train)):
    
    x=x.to(DEVICE)
    y=y.to(DEVICE)
    with torch.no_grad():
        out=CNN_model(x)
    train_data.append(out)
    train_label.append(y)
train_data=torch.cat(train_data)
train_label=torch.cat(train_label)

test_data=[]
test_label=[]
for its,(x,y) in enumerate(tqdm(data_loader_test)):
    
    x=x.to(DEVICE)
    y=y.to(DEVICE)
    with torch.no_grad():
        out=CNN_model(x)
    test_data.append(out)
    test_label.append(y)
test_data=torch.cat(test_data)
test_label=torch.cat(test_label)

torch.save(train_data,'data/mnist/train_data.pkl')
torch.save(train_label,'data/mnist/train_label.pkl')
torch.save(test_data,'data/mnist/test_data.pkl')
torch.save(test_label,'data/mnist/test_label.pkl')


