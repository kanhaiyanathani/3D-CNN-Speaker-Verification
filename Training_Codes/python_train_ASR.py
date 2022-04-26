import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import visdom
from PIL import Image
import cv2
from numpy import linalg as LA
import os
import h5py 

u=os.listdir('/home/SharedData/Royston/Data_ASR_final/train_new_80_2')
dict1={}
for idx in range(0,20000):
	for i in u:
		if(int(i.split("_")[0])==idx+1):
			dict1[idx]=i
print("Dictionary 1 Created")

v=os.listdir('/home/SharedData/Royston/Data_ASR_final/val_new_80_2')
dict2={}

for idx in range(0,4034):
	for i in v:
		if(int(i.split("_")[0])==idx+1):
			dict2[idx]=i
print("Dictionary 2 Created")
			


class voxel_h5(Dataset):
    
	def __init__(self,transform=None):
		self.k23=0

    	def __len__(self):
		return (20000)

    	def __getitem__(self, idx):

		f1 = h5py.File('/home/SharedData/Royston/Data_ASR_final/train_new_80_2/'+dict1[idx],'r') 
		self.data = np.array(f1['0'])
		self.k_0=np.zeros((1,20,80,40))
		self.k_0[0,:,:,:]=self.data
		self.label = (int(dict1[idx].split("_")[1])-1)

    		return self.k_0,self.label

class voxel_h5_val(Dataset):
    
    	def __init__(self,transform=None):
		self.k23=0

   	def __len__(self):
        	return (4034)

    	def __getitem__(self, idx):

		f1 = h5py.File('/home/SharedData/Royston/Data_ASR_final/val_new_80_2/'+dict2[idx],'r') 
		self.data = np.array(f1['0'])
		self.k_0=np.zeros((1,20,80,40))
		self.k_0[0,:,:,:]=self.data
		self.label = (int(dict2[idx].split("_")[1])-1)

    		return self.k_0,self.label


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
	
	self.input_bn= nn.BatchNorm3d(1)
	
	


	self.drp_1 = nn.Dropout3d(p=0.1)

	self.conv1_1 = nn.Conv3d(1, 16, (3, 1, 5), stride=(1, 1, 1))
	self.input_bn1_1=nn.BatchNorm3d(16)
	self.pr1_1 = nn.PReLU()


	self.conv1_2 = nn.Conv3d(16, 16, (3, 9, 1), stride=(1, 2, 1))
	self.input_bn1_2=nn.BatchNorm3d(16)
	self.pr1_2 = nn.PReLU()	

	self.pool1 = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))
	
	self.drp_2 = nn.Dropout3d(p=0.2)

	self.conv2_1 = nn.Conv3d(16, 32, (3, 1, 4), stride=(1, 1, 1))
	self.input_bn2_1=nn.BatchNorm3d(32)
	self.pr2_1 = nn.PReLU()	

	self.conv2_2 = nn.Conv3d(32, 32, (3, 8, 1), stride=(1, 2, 1))
	self.input_bn2_2=nn.BatchNorm3d(32)
	self.pr2_2 = nn.PReLU()	

	self.pool2 = nn.MaxPool3d((1, 1, 2), stride=(1, 1, 2))

	self.drp_3 = nn.Dropout3d(p=0.5)

	self.conv3_1 = nn.Conv3d(32, 64, (3, 1, 3), stride=(1, 1, 1))
	self.input_bn3_1=nn.BatchNorm3d(64)
	self.pr3_1 = nn.PReLU()	

	self.conv3_2 = nn.Conv3d(64, 64, (3, 7, 1), stride=(1, 1, 1))
	self.input_bn3_2=nn.BatchNorm3d(64)
	self.pr3_2 = nn.PReLU()	
	
	self.drp_fc = nn.Dropout(p=0.5)
	self.fc5 = nn.Linear(8*9*5*64,64) #128
	self.bn_fc5=nn.BatchNorm1d(64)		
	self.pr5 = nn.PReLU()


	self.fc6 = nn.Linear(64,200) #128
	self.bn_fc6=nn.BatchNorm1d(200)		
	self.pr6 = nn.PReLU()	


	self.soft= nn.Softmax()

    def forward(self, x):
	
	x=self.input_bn(x)	

	self.y1=self.pr1_1(self.input_bn1_1(self.conv1_1(x)))

	self.y2=self.pr1_2(self.input_bn1_2(self.conv1_2(self.y1)))

	self.y3=self.pool1(self.y2)

	self.y4=self.pr2_1(self.input_bn2_1(self.conv2_1(self.y3)))

	self.y5=self.pr2_2(self.input_bn2_2(self.conv2_2(self.y4)))

	self.y6=self.pool2(self.y5)

	self.y7=self.pr3_1(self.input_bn3_1(self.conv3_1(self.y6)))

	self.y8=self.pr3_2(self.input_bn3_2(self.conv3_2(self.y7)))

	self.y11=self.y8.view(-1,self.num_flat_features(self.y8))

	self.y12=self.bn_fc5(self.pr5(self.fc5(self.y11)))

	self.y13=self.fc6(self.y12)
	
	self.y14=(self.soft(self.y13))

		
        return self.y14

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

net=net.cuda()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
transformed_dataset = voxel_h5()
transformed_dataset2 = voxel_h5_val()

trainloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=128,
                                          shuffle=True, num_workers=8)
evalloader=trainloader

vali_loader = torch.utils.data.DataLoader(transformed_dataset2, batch_size=128,
                                          shuffle=True, num_workers=8)


acc_sum=0
epoch=1
while(True):
    acc=0
    acc_sum=0
    running_loss = 0.0
    epoch=epoch+1
   
    for i, data in enumerate(trainloader, 0):
	net.train(True)
        inputs,label = data
        inputs= Variable(inputs.float().cuda())
	label=Variable(label.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        if i % 80 == 79: 
    		acc=0
    		acc_sum=0		
		for i1, data1 in enumerate(evalloader, 0):  
			net.train(False)
			inputs1,label1 = data1
        		inputs1= Variable(inputs1.float().cuda())
			label1=Variable(label1.cuda())
			outputs1 = net(inputs1)
			predict1 = outputs1.data.max(1)[1]
			acc = predict1.eq(label1.data).cpu().sum()	
			acc_sum += acc

		print('[%d, %5d] Train ACC: %.4f Train acc: %.4f Train total: %.4f' %(epoch + 1, i + 1, float(acc_sum)/20000 ,float(acc_sum),20000 ))
			
		torch.save(net.state_dict(), './convnet_model.pth')
		torch.save(optimizer.state_dict(), './convnet_optimizer.pth')
		

    		acc=0
    		acc_sum=0		
		for i1, data1 in enumerate(vali_loader, 0):  
			#net.train(False)
			inputs1,label1 = data1
        		inputs1= Variable(inputs1.float().cuda())
			label1=Variable(label1.cuda())
			outputs1 = net(inputs1)
			predict1 = outputs1.data.max(1)[1]
			acc = predict1.eq(label1.data).cpu().sum()	
			acc_sum += acc

		print('[%d, %5d] Validation ACC: %.4f Validation acc: %.4f Validation total: %.4f' %(epoch + 1, i + 1, float(acc_sum)/4034 ,float(acc_sum),4034 ))
		print('Model Saved ')

	




