import torch
import torch.nn as nn
from Cla_net import Cla_net
from cov_QFNN import Q_tnorm
# from cov_tri_QFNN import Q_tnorm
# from cov_ti_QFNN import Q_tnorm
from utils import draw_acc, draw_loss, get_logger,setup_seed,acc_cal
from torch.utils import data
import utils as ut

from tqdm import tqdm

tr_batch_size=40
va_batch_size=100
image_type=0
load_cla_model_path='/media/BLACK/COVID_FNN/COV_FNN/result/model/cla_net_10epochs.pth'
EPOCH=100
LR=0.01
DEVICE=torch.device('cuda')
logger=get_logger('cov_QFNN_train')
# setup_seed(777)


cla_net=Cla_net()
cla_net.load_state_dict(torch.load(load_cla_model_path))
cla_net=cla_net.to(DEVICE)
cla_net.eval()

model=Q_tnorm(DEVICE).to(DEVICE)
model_path='result/6qbits_2mems_gossi_all/'
model.load_state_dict(torch.load(model_path+'model/cov_QFNN_0.pth'))

train_data=ut.get_data_train(image_type)
train_loader=data.DataLoader(
    train_data,batch_size=tr_batch_size,shuffle=True,num_workers=2
    )
train_loader_len=len(train_loader)

val_data=ut.get_data_val(image_type)
val_loader=data.DataLoader(
    val_data,batch_size=va_batch_size,shuffle=True,num_workers=2
    )

conunts=train_data.get_num()
logger.critical("train data length:{}".format(len(train_data)))
logger.critical("train data have {} covid,{} normal,{} lung,{} pne".format(
    conunts[0],conunts[1],conunts[2],conunts[3]
    ))

conunts=val_data.get_num()
logger.critical("val data length:{}".format(len(val_data)))
logger.critical("val data have {} covid,{} normal,{} lung,{} pne".format(
    conunts[0],conunts[1],conunts[2],conunts[3]
    ))
# for i in model.named_parameters():
#     print(i)
optimizer=torch.optim.AdamW(model.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

train_loss_list=[]
train_acc_list=[]
val_loss_list=[]
val_acc_list=[]
# train_loss_list=torch.load('result/data/cov_train_loss')
# train_acc_list=torch.load('result/data/cov_train_acc')
# val_loss_list=torch.load('result/data/cov_val_loss')
# val_acc_list=torch.load('result/data/cov_val_acc')

for epoch in range(EPOCH):
    break
    logger.info(f'=======================epoch:{epoch}=======================')
    for its,(x,y) in enumerate(tqdm(train_loader)):
        
        model.train()
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        with torch.no_grad():
            qfnn_in=cla_net(x)
        output=model(qfnn_in)
        loss=loss_func(output,y)
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
            for (x,y) in val_loader:
                step+=1
                x=x.to(DEVICE)
                y=y.to(DEVICE)
                qfnn_in=cla_net(x)
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

    torch.save(model.state_dict(),f'result/model/qfnn/cov_QFNN_{epoch}.pth')
    torch.save(train_loss_list,'result/data/cov_train_loss')
    torch.save(train_acc_list,'result/data/cov_train_acc')
    torch.save(val_loss_list,'result/data/cov_val_loss')
    torch.save(val_acc_list,'result/data/cov_val_acc')
    

test_data=ut.get_data_test(image_type)
test_loader=data.DataLoader(
    test_data,batch_size=va_batch_size,shuffle=True,num_workers=2
    )
test_loader_len=len(test_loader)

conunts=test_data.get_num()
logger.critical("test data length:{}".format(len(test_data)))
logger.critical("test data have {} covid,{} normal,{} lung,{} pne".format(
    conunts[0],conunts[1],conunts[2],conunts[3]
    ))

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
model.eval()
pred=[]
label=[]
with torch.no_grad():
        total=0
        correct=0
        step=0
        for (x,y) in tqdm(test_loader):
            step+=1
            x=x.to(DEVICE)
            y=y.to(DEVICE)
            qfnn_in=cla_net(x)
            output=model(qfnn_in)
            predicted=torch.argmax(output,1)
            pred.extend(predicted.cpu().numpy())
            label.extend(y.cpu().numpy())
            acc=accuracy_score(label,pred)
            precision=precision_score(label,pred,average='macro')
            recall=recall_score(label,pred,average='macro')
            f1=f1_score(label,pred,average='macro')
            logger.info(f'acc:{acc} precision:{precision} recall:{recall} f1:{f1}')
            # total+=y.size(0)
            # correct+=(predicted==y).sum().item()
        # test_acc=correct/total
        # logger.info(f'test_acc:{test_acc}')

# torch.save(pred,model_path+'cov_pred')
# torch.save(label,model_path+'cov_label')

# pred=torch.load(model_path+'cov_pred')
# label=torch.load(model_path+'cov_label')

acc=accuracy_score(label,pred)
precision=precision_score(label,pred,average='macro')
recall=recall_score(label,pred,average='macro')
f1=f1_score(label,pred,average='macro')
logger.info(f'acc:{acc} precision:{precision} recall:{recall} f1:{f1}')
cm=confusion_matrix(label,pred)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.figure(figsize=(10,10))
#改变标签名称
labels_name=['covid','normal','lung','pne']
cm=pd.DataFrame(cm,index=labels_name,columns=labels_name)
#字体大小
sns.set(font_scale=1.5)
#cmap种类： 
sns.heatmap(cm,annot=True,fmt='.20g',cmap='GnBu')
plt.savefig(model_path+'cov_cm.png')

