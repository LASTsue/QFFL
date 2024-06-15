import torch
from tqdm import tqdm
from cov_QFNN import Qfnn
# from cov_QFNN_ti import Qfnn
# from cov_QFNN_tri import Qfnn

from utils import setup_seed

DEVICE=torch.device('cuda')
setup_seed(777)
node=3
# #测试
test_data=torch.load('data/cov/test_data.pth').to(DEVICE)
label=torch.load('data/cov/test_label.pth').to(DEVICE)

data_loader=torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(test_data,label),
                                                batch_size = 5000,
                                                shuffle = True)

NAME='cov_QFNN_gas_q4_star'

gmm_list=torch.load(f'result/data/{NAME}_gmm_list')
data_weights=torch.load(f'result/data/{NAME}_data_weights')
gmm_scores=[]
for i in range(node):
    gmm_scores.append(gmm_list[i].score_samples(test_data.cpu().numpy()))

gmm_scores=torch.tensor(gmm_scores).to(DEVICE).permute(1,0)

for i in range(node):
    gmm_scores[:,i]=gmm_scores[:,i]*data_weights[i]
sum=torch.sum(gmm_scores,dim=1)
for i in range(node):
    gmm_scores[:,i]=gmm_scores[:,i]/sum

out_put=[]


for i in tqdm(range(node)):
    model=Qfnn(DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(f'result/model/qfnn/{NAME}_n{i}.pth'))
    model.eval()
    with torch.no_grad():
        out_put.append(model(test_data))

out_put=torch.stack(out_put,dim=1)
for i in range(node):
    m=out_put[:,i,:]
    n=gmm_scores[:,i].unsqueeze(1)
    out_put[:,i,:]=out_put[:,i,:]*gmm_scores[:,i].unsqueeze(1)
out_put=torch.sum(out_put,dim=1)
out_put=torch.softmax(out_put,dim=1)
pred=torch.argmax(out_put,dim=1)
# acc=torch.sum(predict==test_label)/len(test_label)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
pred=pred.cpu().numpy()
label=label.cpu().numpy()
acc=accuracy_score(label,pred)
precision=precision_score(label,pred,average='macro')
recall=recall_score(label,pred,average='macro')
f1=f1_score(label,pred,average='macro')
print(f'acc:{acc} precision:{precision} recall:{recall} f1:{f1}')
cm=confusion_matrix(label,pred)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# plt.figure(figsize=(10,10))
# #改变标签名称
# labels_name=['covid','normal','lung','pne']
# cm=pd.DataFrame(cm,index=labels_name,columns=labels_name)
# #字体大小
# sns.set(font_scale=1.5)
# #cmap种类： 
# sns.heatmap(cm,annot=True,fmt='.20g',cmap='GnBu')
# plt.savefig('cov_cm.png')

# print(acc)


