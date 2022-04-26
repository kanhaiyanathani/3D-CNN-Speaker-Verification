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
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
from pylab import *


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

	self.y13=(self.fc6(self.y12))
	
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

net.load_state_dict(torch.load('./convnet_model.pth'))
optimizer.load_state_dict(torch.load('./convnet_optimizer.pth'))
net.train(False)
def get_feature(filename):
	f1 = h5py.File(filename,'r') 
	data = np.array(f1['0'])
	b=[]
	#rep=2
	data=np.repeat(np.array(f1['0']),rep,axis=0)
	for i in range(rep):
		d = data[20*i:20*(i+1)]
		#d1=np.tile(d,(20,1,1))
		k_0=np.zeros((1,1,20,80,40))
		k_0[0,:,:,:]=d

		test_2=torch.from_numpy(k_0)
		test_2=Variable(test_2.float().cuda())

		outputs1 = net(test_2)
		b.append((((net.y12).data).cpu().numpy())[0])
	return b
rep=1
data=[]
[data.append([]) for _ in xrange(32)]
count=1
for entry in os.listdir('/home/SharedData/Royston/Data_ASR_final/train_new_100_2/'):
	data[int(entry.split('_')[1])-1].extend(get_feature('/home/SharedData/Royston/Data_ASR_final/train_new_100_2/'+entry))
	#print(count)
	count+=1

for i in xrange(32):
	data[i]=np.array(data[i])	
np.save("data_feature_new.npy",data)

data=np.load('data_feature_new.npy')
confusion=np.zeros((32,32))
for k in range(32):
	for j in np.argmin([(cdist(data[i][:80*rep],data[k][80*rep:])).min(axis=0) for i in range(32)],axis=0):
		confusion[k,j]+=1
print(" SER : %s " %((1- confusion.trace()/20.0/rep/32.0)*100))
np.savetxt('./confusion_new.txt',confusion, fmt='%d')
print(confusion)

pos_data=[]
for i in range(32):
	pos_data.extend(data[i][80*rep:])

neg_data=[]
#[data.append([]) for _ in xrange(32)]
count1=1
for entry in os.listdir('/home/SharedData/Royston/Data_ASR_final/test_bar_640/'):
	neg_data.extend(get_feature('/home/SharedData/Royston/Data_ASR_final/test_bar_640/'+entry))
	#print(count)
	count1+=1

total_data=np.concatenate((pos_data,neg_data[:640]),axis=0)
y_label=[0]*len(pos_data)+[1]*640

y_pred = np.array([(cdist(data[i][:80],total_data)).min(axis=0) for i in range(32)]).min(axis=0)
fpr, tpr, threshold = roc_curve(y_label, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.argmin(np.absolute((fnr - fpr)))]
EER = fpr[np.argmin(np.absolute((fnr - fpr)))]
print(EER)

plot(fpr,tpr,label='ROC Curve')
plot(x,y,label='x+y=1 line')
legend(loc='upper right')
title('ROC curve when train codebook')
ylabel(' True Positive Rate--->')
xlabel(' False Positive Rate--->')
show()
'''
y_pred = np.sum([(cdist(data[i][:80],total_data)).min(axis=0) for i in range(32)],0)
fpr, tpr, threshold = roc_curve(y_label, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.argmin(np.absolute((fnr - fpr)))]
EER = fpr[np.argmin(np.absolute((fnr - fpr)))]
print(EER)

y_pred = np.array([np.sum(cdist(data[i][:80],total_data),0) for i in range(32)]).min(axis=0)
fpr, tpr, threshold = roc_curve(y_label, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.argmin(np.absolute((fnr - fpr)))]
EER = fpr[np.argmin(np.absolute((fnr - fpr)))]
print(EER)

y_pred = np.sum([np.sum(cdist(data[i][:80],total_data),0) for i in range(32)],0)

fpr, tpr, threshold = roc_curve(y_label, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.argmin(np.absolute((fnr - fpr)))]
EER = fpr[np.argmin(np.absolute((fnr - fpr)))]
print(EER)
'''


