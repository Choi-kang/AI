import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np

#使用torchvision加载并预处理CIFAR10数据集
show = ToPILImage()         #可以把Tensor转成Image,方便进行可视化
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])#把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
trainset = tv.datasets.CIFAR10(root='data1/',train = True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
testset = tv.datasets.CIFAR10('data1/',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=0)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
(data,label) = trainset[100]
print(classes[label])#输出ship
show((data+1)/2).resize((100,100))
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400,100))#make_grid的作用是将若干幅图像拼成一幅图像

#定义网络
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return  x

net = Net()
print(net)

rnn=[0.003,0.0003,0.00003]
losst=[[]for i in range(3)]
j=0
torch.save(net,'mo.pth')
for i in rnn:
    
    #定义损失函数和优化器
    from torch import optim
    net=torch.load('mo.pth')
    criterion  = nn.CrossEntropyLoss()#定义交叉熵损失函数
    optimizer = optim.SGD(net.parameters(),lr = i,momentum=0.9)

    #训练网络
    from torch.autograd  import Variable
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):#enumerate将其组成一个索引序列，利用它可以同时获得索引和值,enumerate还可以接收第二个参数，用于指定索引起始值
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss  = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 ==1999:
                print('[%d, %5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
                losst[j].append(running_loss/30)
                running_loss = 0.0
    print("----------finished training---------")
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    print('实际的label: ',' '.join('%08s'%classes[labels[j]] for j in range(4)))
    show(tv.utils.make_grid(images/2 - 0.5)).resize((400,100))#？？？？？
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data,1)#返回最大值和其索引
    print('预测结果:',' '.join('%5s'%classes[predicted[j]] for j in range(4)))
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total +=labels.size(0)
        correct +=(predicted == labels).sum()
    print('10000张测试集中的准确率为: %d %%'%(100*correct/total))
    j=j+1
print(losst)
f1=plt.figure(5)#弹出对话框时的标题，如果显示的形式为弹出对话框的话
plt.subplot(221).set(title='Learn_rate=0.003')
plt.subplot(222).set(title='Learn_rate=0.0003')
plt.subplot(212).set(title='Learn_rate=0.00003')
xdata = ['0','2000','4000','6000','8000','10000']
plt.subplot(221).plot(xdata,losst[0])
plt.subplot(222).plot(xdata,losst[1])
plt.subplot(212).plot(xdata,losst[2])
plt.subplots_adjust(left=0.08,right=0.95,wspace=0.25,hspace=0.45)
plt.show()
xd1=['0','2000','4000','6000','8000','10000']
yd1=losst[0]
yd2=losst[1]
yd3=losst[2]
plt.plot(xd1,yd1,label="0.003",color="red")
plt.plot(xd1,yd2,label="0.0003",color="blue")
plt.plot(xd1,yd3,label="0.00003",color="green")
plt.xlabel("Data_amount")
plt.ylabel("Loss")
plt.title("Learn_rate")
#在ipython的交互环境中需要这句话才能显示出来
plt.show()