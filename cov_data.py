from torch.utils import data
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms

class Z_Data(data.Dataset):
    def __init__(self,data_type,img_type=0) -> None:
        super().__init__()
        #0:raw 1:masks
        if img_type==0:
            self.img_type_path='images/'
        elif img_type==1:
            self.img_type_path='masks/'
        else:
            print('img type error')
            exit()
        self.path='/media/BLACK/COVID_FNN/COV_FNN/datasets/COVID-19/'
        if data_type=='train':
            self.data=pd.read_csv('/media/BLACK/COVID_FNN/COV_FNN/datasets/COVID-19/train.csv')
        elif data_type=='val':
            self.data=pd.read_csv('/media/BLACK/COVID_FNN/COV_FNN/datasets/COVID-19/val.csv')
        elif data_type=='test':
            self.data=pd.read_csv('/media/BLACK/COVID_FNN/COV_FNN/datasets/COVID-19/test.csv')
        else:
            print('data type error')
            exit()
        
        
        
    def __getitem__(self, index):
        label=self.data.loc[index,'LABEL']
        if label==0:
            img_path=self.path+'COVID/'+self.img_type_path+self.data.loc[index,'FILE NAME']+'.png'
        elif label==1:
            img_path=self.path+'Normal/'+self.img_type_path+self.data.loc[index,'FILE NAME']+'.png'
        elif label==2:
            img_path=self.path+'Lung_Opacity/'+self.img_type_path+self.data.loc[index,'FILE NAME']+'.png'
        elif label==3:
            img_path=self.path+'Viral Pneumonia/'+self.img_type_path+self.data.loc[index,'FILE NAME']+'.png'
        else:
            print('data label error')
            exit()
        img=Image.open(img_path).convert('RGB')
        img=transforms.ToTensor()(img)
        return img,label
    
    def __len__(self):
        return len(self.data)
    
    def get_num(self):

        cov_num=0
        normal_num=0
        lung_num=0
        pne_num=0

        for i in range(len(self.data)):
            if self.data.loc[i,'LABEL']==0:
                cov_num+=1
            elif self.data.loc[i,'LABEL']==1:
                normal_num+=1
            elif self.data.loc[i,'LABEL']==2:
                lung_num+=1
            elif self.data.loc[i,'LABEL']==3:
                pne_num+=1
            else:
                print('data label error')
                exit()
        return (cov_num,normal_num,lung_num,pne_num)
    
# if __name__=='__main__':
#     datas=Z_Data('train',0)
#     dataLoader=data.DataLoader(datas,batch_size=100,shuffle=True,num_workers=2)
#     for img,label in dataLoader:
#         print(img.shape)
        