import torch
import torch.nn as nn
from Cla_net import Cla_net

from utils import draw_acc, draw_loss, get_logger,setup_seed,acc_cal
from torch.utils import data
import utils as ut

from tqdm import tqdm

tr_batch_size=40
image_type=0
load_cla_model_path='/media/BLACK/COVID_FNN/COV_FNN/result/model/cla_net_10epochs.pth'

DEVICE=torch.device('cuda')



cla_net=Cla_net()
cla_net.load_state_dict(torch.load(load_cla_model_path))
cla_net=cla_net.to(DEVICE)
cla_net.eval()


train_data=ut.get_data_train(image_type)
train_loader=data.DataLoader(
    train_data,batch_size=tr_batch_size,shuffle=True,num_workers=2
    )
train_loader_len=len(train_loader)

train_data_list=[]
train_label_list=[]
for its,(x,y) in enumerate(tqdm(train_loader)):
    x=x.to(DEVICE)
    y=y.to(DEVICE)
    with torch.no_grad():
        out=cla_net(x)
    train_data_list.append(out)
    train_label_list.append(y)
train_data=torch.cat(train_data_list)
train_label=torch.cat(train_label_list)

torch.save(train_data,'data/cov/cov_train_data.pth')
torch.save(train_label,'data/cov/cov_train_label.pth')

test_data=ut.get_data_test(image_type)
test_loader=data.DataLoader(
    test_data,batch_size=tr_batch_size,shuffle=True,num_workers=2
    )
test_loader_len=len(test_loader)

test_data_list=[]
test_label_list=[]

for its,(x,y) in enumerate(tqdm(test_loader)):
    x=x.to(DEVICE)
    y=y.to(DEVICE)
    with torch.no_grad():
        out=cla_net(x)
    test_data_list.append(out)
    test_label_list.append(y)

test_data=torch.cat(test_data_list)
test_label=torch.cat(test_label_list)

torch.save(test_data,'data/cov/cov_test_data.pth')
torch.save(test_label,'data/cov/cov_test_label.pth')


