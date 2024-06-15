import torch
import torch.nn as nn

class Cla_net(torch.nn.Module):
    def __init__(self) -> None:
        super(Cla_net,self).__init__()
        self.conv1=nn.Conv2d(3,64,3,padding=1)
        self.conv2=nn.Conv2d(64,64,3,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.mp1=nn.MaxPool2d(2)

        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.conv4=nn.Conv2d(128,128,3,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.mp2=nn.MaxPool2d(2)

        self.conv5=nn.Conv2d(128,256,3,padding=1)
        self.conv6=nn.Conv2d(256,256,3,padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.mp3=nn.MaxPool2d(2)

        self.conv7=nn.Conv2d(256,512,3,padding=1)
        self.conv8=nn.Conv2d(512,512,3,padding=1)
        self.bn4=nn.BatchNorm2d(512)
        self.mp4=nn.MaxPool2d(2)

        self.avgpool=nn.AdaptiveAvgPool2d((5,5))
        self.lstm1=nn.LSTM(25,100,2,batch_first=True,dropout=0.5)
        self.dr1=nn.Dropout(0.5)
        self.lstm2=nn.LSTM(100,50,2,batch_first=True,dropout=0.5)
        self.dr2=nn.Dropout(0.5)
        self.bn5=nn.BatchNorm1d(50*512)
        self.dr3=nn.Dropout(0.5)
        self.linear1=nn.Linear(50*512,4)

    
    def forward(self,input):
        x=self.conv1(input)
        x=torch.relu(x)
        x=self.conv2(x)
        x=torch.relu(x)
        x=self.bn1(x)
        x=self.mp1(x)

        x=self.conv3(x)
        x=torch.relu(x)
        x=self.conv4(x)
        x=torch.relu(x)
        x=self.bn2(x)
        x=self.mp2(x)

        x=self.conv5(x)
        x=torch.relu(x)
        x=self.conv6(x)
        x=torch.relu(x)
        x=self.bn3(x)
        x=self.mp3(x)

        x=self.conv7(x)
        x=torch.relu(x)
        x=self.conv8(x)
        x=torch.relu(x)
        x=self.bn4(x)
        x=self.mp4(x)
        
        x=self.avgpool(x)
        x=x.view(-1,512,25)
        x=self.dr1(x)
        x,_=self.lstm1(x)
        x=torch.relu(x)
        x=self.dr2(x)
        x,_=self.lstm2(x)
        x=torch.relu(x)
        x=x.reshape(-1,50*512)
        x=self.bn5(x)
        x=self.dr3(x)
        x=self.linear1(x)
        return x


#main
# if __name__ == "__main__":
#     data=torch.randn(10,3,299,299)
#     cla_net=Cla_net()
#     cla_net=nn.Sequential(*list(cla_net.children())[:-2])
#     print(cla_net)