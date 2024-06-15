import torch
import torch.nn as nn
from mni_QFNN import Qfnn
import numpy as np
# from mni_QFNN_ti import Qfnn
# from mni_QFNN_tri import Qfnn
from utils import get_logger,acc_cal,setup_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture
from opacus import PrivacyEngine
from tqdm import tqdm
from opacus.utils.batch_memory_manager import BatchMemoryManager


BATCH_SIZE=600
EPOCH=5
LR=0.1
DEVICE=torch.device('cuda')



setup_seed(777)

transform = transforms.Compose([transforms.ToTensor()])

train_data=torch.load('data/mnist/train_data.pkl').cpu().numpy()
train_label=torch.load('data/mnist/train_label.pkl').cpu().numpy()

all_len=len(train_label)

gmm_list=[]
weights=[]

NAME=f'mnist_QFNN_gas_q4_star'
node=9

# keep_list = [[(i+j)%10 for j in range(5)] for i in range(10)]
keep_list = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9]]
for i in range(node):

    model=Qfnn(DEVICE).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR)
    loss_func=nn.CrossEntropyLoss()

    train_loss_list=[]
    train_acc_list=[]

    keep = np.isin(train_label, keep_list[i])
    data=np.array(train_data)[keep]
    labels=np.array(train_label)[keep]

    weights.append(len(data)/all_len)


    gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
    gmm.fit(data)
    gmm_list.append(gmm)
    #打包数据和标签
    train_data_set=torch.utils.data.TensorDataset(torch.tensor(data),torch.tensor(labels))
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






    
