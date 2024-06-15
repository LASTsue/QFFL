import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cov_QFNN import Qfnn
# from cov_QFNN_ti import Qfnn
# from cov_QFNN_tri import Qfnn
from utils import get_logger,acc_cal,setup_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture

from tqdm import tqdm



BATCH_SIZE=64
VAL_BATCH_SIZE=6000
EPOCH=5
LR=0.01
DEVICE=torch.device('cuda')
NAME='cov_QFNN_gas_q4_ring3'
node=3

setup_seed(777)

transform = transforms.Compose([transforms.ToTensor()])

train_data=torch.load('data/cov/train_data.pth').to(DEVICE)
train_label=torch.load('data/cov/train_label.pth').to(DEVICE)

all_len=len(train_label)

gmm_list=[]
weights=[]

# keep_list=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
keep_list = [[(i+j)%4 for j in range(3)] for i in range(4)]
for i in range(node):

    model=Qfnn(DEVICE).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR)
    loss_func=nn.CrossEntropyLoss()

    train_loss_list=[]
    train_acc_list=[]

    keep = (train_label == keep_list[i][0]) | (train_label == keep_list[i][1]) 
    data=train_data[i*(all_len//node):(i+1)*(all_len//node)]
    labels=train_label[i*(all_len//node):(i+1)*(all_len//node)]

    weights.append(len(data)/all_len)


    gmm = GaussianMixture(n_components=6, max_iter=500, random_state=777)
    gmm.fit(data.cpu().numpy())
    gmm_list.append(gmm)
    #打包数据和标签
    train_data_set=torch.utils.data.TensorDataset(data,labels)
    train_data_loader=torch.utils.data.DataLoader(dataset=train_data_set,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True)

    for epoch in range(EPOCH):
        print(f'=======================node:{i}  epoch:{epoch}=======================')
       
        for its,(x,y) in enumerate(tqdm(train_data_loader)):

            model.train()
            x=x.to(DEVICE)
            y=y.to(DEVICE)
            output=model(x)
            loss=loss_func(output,y)
            train_loss_list.append(loss.item())
            acc=acc_cal(output,y)
            train_acc_list.append(acc)
            tqdm.write(f'loss:{loss.item()}  acc:{acc}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    torch.save(model.state_dict(),f'result/model/qfnn/{NAME}_n{i}.pth')
    torch.save(train_loss_list,f'result/data/{NAME}_train_loss_n{i}')
    torch.save(train_acc_list,f'result/data/{NAME}_train_acc_n{i}')

torch.save(gmm_list,f'result/data/{NAME}_gmm_list')
torch.save(weights,f'result/data/{NAME}_data_weights')







    
