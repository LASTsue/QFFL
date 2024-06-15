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


BATCH_SIZE=512
VAL_BATCH_SIZE=6000
EPOCH=100
LR=0.01
DEVICE=torch.device('cuda')
logger=get_logger('QFNN_train')
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

#划分验证集
train_size = int(0.9 * len(data_train))
val_size = len(data_train) - train_size
data_train, data_val = torch.utils.data.random_split(data_train, [train_size, val_size])


data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                 num_workers=2)

data_loader_val = torch.utils.data.DataLoader(dataset=data_val,
                                                batch_size = VAL_BATCH_SIZE,
                                                shuffle = True,
                                                 num_workers=2)

counts=[0]*10
for (x,y) in data_loader_train:
    for i in y:
        counts[i]+=1
data_len=len(data_train)
logger.info(f'train data have {data_len} samples')
logger.info(f'train data have {counts} samples')

counts=[0]*10
for (x,y) in data_loader_val:
    for i in y:
        counts[i]+=1
data_len=len(data_val)
logger.info(f'val data have {data_len} samples')
logger.info(f'val data have {counts} samples')



CNN_model=torch.load('result/model/CNN.pkl')
CNN_model=CNN_model.to(DEVICE)
CNN_model.eval()
# model_path='result/6qbits_2mems_gossi_all/'
model=Qfnn(DEVICE).to(DEVICE)
# model.load_state_dict(torch.load(model_path+'model/QFNN_0.pth'))

# for i in model.named_parameters():
#     print(i)
optimizer=torch.optim.AdamW(model.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

train_loss_list=[]
train_acc_list=[]
val_loss_list=[]
val_acc_list=[]

for epoch in range(EPOCH):
    logger.info(f'=======================epoch:{epoch}=======================')
    for its,(x,y) in enumerate(tqdm(data_loader_train)):
        
        model.train()
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        with torch.no_grad():
            qfnn_in=CNN_model(x)
        output=model(qfnn_in)
        loss=loss_func(output,y)
        train_loss_list.append(loss.item())
        acc=acc_cal(output,y)
        train_acc_list.append(acc)
        # logger.info(f'train_loss:{loss.item()} train_acc:{acc}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        draw_loss(train_loss_list,'train_loss')
        draw_acc(train_acc_list,'train_acc')

        with torch.no_grad():
            total=0
            correct=0
            step=0
            loss_val=0
            for (x,y) in data_loader_val:
                step+=1
                x=x.to(DEVICE)
                y=y.to(DEVICE)
                qfnn_in=CNN_model(x)
                output=model(qfnn_in)
                loss=loss_func(output,y)
                loss_val+=loss.item()
                predicted=torch.argmax(output,1)
                total+=y.size(0)
                correct+=(predicted==y).sum().item()
            val_acc=correct/total
            val_loss=loss_val/step
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            # logger.info(f'val_loss:{val_loss} val_acc:{val_acc}')
            draw_loss(val_loss_list,'val_loss')
            draw_acc(val_acc_list,'val_acc')
    
    torch.save(model.state_dict(),f'result/model/qfnn/mnist_QFNN_{epoch}.pth')
    torch.save(train_loss_list,'result/data/mnist_train_loss')
    torch.save(train_acc_list,'result/data/mnist_train_acc')
    torch.save(val_loss_list,'result/data/mnist_val_loss')
    torch.save(val_acc_list,'result/data/mnist_val_acc')
    
# model.eval()

# data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
#                                                batch_size = 5000,
#                                                shuffle = True,
#                                                 num_workers=2)
# counts=[0]*10
# for (x,y) in data_loader_test:
#     for i in y:
#         counts[i]+=1

# data_len=len(data_test)
# logger.info(f'test data have {data_len} samples')
# logger.info(f'test data have {counts} samples')

# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
# with torch.no_grad():
#     total=0
#     correct=0
#     pred=[]
#     label=[]
#     for step,(x,y) in enumerate(tqdm(data_loader_test)):
#         x=x.to(DEVICE)
#         y=y.to(DEVICE)
#         qfnn_in=CNN_model(x)
#         output=model(qfnn_in)
#         predicted=torch.argmax(output,1)
#         #extend:将另一个集合中的元素逐一添加到列表中
#         pred.extend(predicted.cpu().numpy())
#         label.extend(y.cpu().numpy())
#         acc=accuracy_score(label,pred)
#         precision=precision_score(label,pred,average='macro')
#         recall=recall_score(label,pred,average='macro')
#         f1=f1_score(label,pred,average='macro')
#         cm=confusion_matrix(label,pred)
#         logger.info(f'test_acc:{acc} test_precision:{precision} test_recall:{recall} test_f1:{f1}')
        # total+=y.size(0)
        # correct+=(predicted==y).sum().item()
    # acc=correct/total
    # logger.info(f'test_acc:{acc}')
# torch.save(pred,model_path+'mnist_pred')
# torch.save(label,model_path+'mnist_label')
# acc=accuracy_score(label,pred)
# precision=precision_score(label,pred,average='macro')
# recall=recall_score(label,pred,average='macro')
# f1=f1_score(label,pred,average='macro')
# cm=confusion_matrix(label,pred)
# logger.info(f'test_acc:{acc} test_precision:{precision} test_recall:{recall} test_f1:{f1}')
#绘制混淆矩阵
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(cm,annot=True,fmt='.20g',cmap='Blues')
# plt.savefig(model_path+'mnist_cm.png')

    
