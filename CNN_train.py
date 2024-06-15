import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CNN import CNN
from utils import get_logger,setup_seed,draw_loss,draw_acc,acc_cal
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import time


BATCH_SIZE=5000
EPOCH=50
LR=0.001
DEVICE=torch.device('cuda')
logger=get_logger('CNN_train')
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
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                 num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = BATCH_SIZE,
                                               shuffle = True,
                                                num_workers=2)

model=CNN().to(DEVICE)
model=torch.load('result/model/CNN.pkl')
optimizer=torch.optim.AdamW(model.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

loss_list=[]
acc_list=[]

for epoch in range(EPOCH):

    model.train()
    for its,(x,y) in enumerate(tqdm(data_loader_train)):
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        output=model(x)
        loss=loss_func(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        acc_list.append(acc_cal(output,y))
    logger.info(f'epoch:{epoch} train_loss:{loss.item()}')
    torch.save(model,f'result/model/CNN_{epoch}.pkl')
    
    model.eval()
    with torch.no_grad():
        total=0
        correct=0
        for step,(x,y) in enumerate(tqdm(data_loader_test)):
            x=x.to(DEVICE)
            y=y.to(DEVICE)
            output=model(x)
            predicted=torch.argmax(output,1)
            total+=y.size(0)
            correct+=(predicted==y).sum().item()
        acc=correct/total
        logger.info(f'epoch:{epoch} test_acc:{acc}')
