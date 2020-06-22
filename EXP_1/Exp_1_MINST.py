import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import os.path

#参数设置
DOWNLOAD_DATASET = False #是否下载数据集
BATCH_SIZE = 64 #Batch_Size
SHUFFLE = True #是否打乱数据集
LEARNING_RATE = 0.001 #学习率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#是否使用GPU
EPOCH = 1 #Epoch
PRETRAIN = True #是否加载预训练模型
MODULE = 'AlexNet' #使用的网络 LeNet5, AlexNet
PATH = './module/' #模型的保存路径
OPTIM = 'Adam' #模型的优化方式 Adam or SGD
MOMENTUM = 0.9 #SGD动态率

#数据集下载
train_dataset = datasets.MNIST(root='./dataset/', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_DATASET)
test_dataset = datasets.MNIST(root='./dataset/', train=False, transform=transforms.ToTensor(), download=DOWNLOAD_DATASET)

#数据集载入
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#MNIST数据可视化
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1, 2, 0)
print(labels)
cv2.imshow('MNIST_DATA', img)
key_pressed = cv2.waitKey(0)

#网络实现
if MODULE == 'LeNet5':
	# LeNet5
	class LeNet5(nn.Module):
		def __init__(self):
			super(LeNet5, self).__init__()
			self.cov1 = nn.Sequential(  # INPUT(1, 28, 28) OUTPUT(6, 14, 14)
				nn.Conv2d(  # 卷积层 data(1, 28, 28)
					in_channels=1,  # 定义图片的高度(多少高度层)
					out_channels=6,  # filter输出的高度，这里指6个filter同时进行扫描操作
					kernel_size=5,  # filter的宽和高都是5
					stride=1,  # 每次滑动的位移量
					padding=2,  # 图片周围围2圈0，这里计算为2是根据kernel_size还有data_size决定的
					# padding=(kernel_size-1)/2=(5-1)/2
				),
				nn.ReLU(),  # 激活函数#data(6,28,28)
				nn.MaxPool2d(kernel_size=2)  # 池化层 #data(6,14,14)
			)

			self.cov2 = nn.Sequential(  # INPUT(6, 14, 14) OUTPUT(16, 5, 5)
				nn.Conv2d(  # 卷积层 data(6, 14, 14)
					in_channels=6,
					out_channels=16,
					kernel_size=5,
					stride=1,
				),
				nn.ReLU(),  # 激活函数#data(16,10,10)
				nn.MaxPool2d(kernel_size=2)  # 池化层 #data(16,5,5)
			)

			self.fc1 = nn.Sequential(  # 全连接层 #INPUT(16*5*5) OUTPUT(120)
				nn.Linear(16 * 5 * 5, 120),
				nn.BatchNorm1d(120),
				nn.ReLU()
			)
			self.fc2 = nn.Sequential(  # 全连接层 #INPUT(120) OUTPUT(10)
				nn.Linear(120, 84),
				nn.BatchNorm1d(84),
				nn.ReLU(),
				nn.Linear(84, 10)
			)

		def forward(self, x):
			x = self.cov1(x)  # 卷积->激活->池化
			x = self.cov2(x)  # 卷积->激活->池化
			x = x.reshape(x.size()[0], -1)  # 一维展开
			x = self.fc1(x)  # 全连接->BN->激活
			x = self.fc2(x)  # 全连接->BN->激活->全连接
			return x


	# 构建网络
	Net5 = LeNet5().to(DEVICE)
	# Loss使用CrossEntropy
	Loss_Function = nn.CrossEntropyLoss()
	if OPTIM == 'Adam':
		# Optim使用Adam
		Optim = optim.Adam(
			Net5.parameters(),
			lr=LEARNING_RATE,
		)
	else:
		Optim = optim.SGD(
			Net5.parameters(),
			lr=LEARNING_RATE,
			momentum=MOMENTUM
		)
if MODULE == 'AlexNet':
	# AlexNet
	class AlexNet(nn.Module):
		def __init__(self):
			super(AlexNet, self).__init__()
			self.cov1 = nn.Sequential(  # INPUT(1, 28, 28) OUTPUT(32, 14, 14)
				nn.Conv2d(  # 卷积层 data(1, 28, 28)
					in_channels=1,  # 定义图片的高度(多少高度层)
					out_channels=32,  # filter输出的高度，这里指32个filter同时进行扫描操作
					kernel_size=3,  # filter的宽和高都是3
					stride=1,  # 每次滑动的位移量
					padding=1,  # 图片周围围1圈0，这里计算为1是根据kernel_size还有data_size决定的
				),
				nn.ReLU(),  # 激活函数#data(32, 28, 28)
				nn.MaxPool2d(kernel_size=2, stride=2) # 池化层 #data(32, 14, 14)
			)

			self.cov2 = nn.Sequential(  # INPUT(32, 14, 14) OUTPUT(64, 7, 7)
				nn.Conv2d(  # 卷积层 data(32, 14, 14)
					in_channels=32,
					out_channels=64,
					kernel_size=3,
					stride=1,
					padding=1
				),
				nn.ReLU(),  # 激活函数#data(64, 28, 28)
				nn.MaxPool2d(kernel_size=2, stride = 2)  # 池化层 #data(64, 7, 7)
			)
			self.cov3 = nn.Sequential(  # INPUT(64, 7, 7) OUTPUT(256, 3, 3)
				nn.Conv2d(  # 卷积层
					in_channels=64,
					out_channels=128,
					kernel_size=3,
					stride=1,
					padding=1
				),
				nn.ReLU(),  # 激活函数
				nn.Conv2d(  # 卷积层
					in_channels=128,
					out_channels=256,
					kernel_size=3,
					stride=1,
					padding=1
				),
				nn.ReLU(),  # 激活函数
				nn.Conv2d(  # 卷积层
					in_channels=256,
					out_channels=256,
					kernel_size=3,
					stride=1,
					padding=1
				),
				nn.ReLU(),  # 激活函数
				nn.MaxPool2d(kernel_size=2, stride = 2),  # 池化层
			)
			self.fc = nn.Sequential(  # 全连接层 #INPUT(256*3*3) OUTPUT(10)
				nn.Linear(256*3*3, 1024),
				nn.BatchNorm1d(1024),
				nn.ReLU(),
				nn.Dropout(0.5),
				nn.Linear(1024, 512),
				nn.BatchNorm1d(512),
				nn.ReLU(),
				nn.Dropout(0.5),
				nn.Linear(512, 10),
			)


		def forward(self, x):
			x = self.cov1(x)  # 卷积->激活->池化
			x = self.cov2(x)  # 卷积->激活->池化
			x = self.cov3(x)  # 卷积->激活->池化
			x = x.reshape(x.size()[0], -1)  # 一维展开
			x = self.fc(x)  # 全连接->BN->激活
			return x


	# 构建网络
	Alex = AlexNet().to(DEVICE)
	# Loss使用CrossEntropy
	Loss_Function = nn.CrossEntropyLoss()
	if OPTIM == 'Adam':
		# Optim使用Adam
		Optim = optim.Adam(
			Alex.parameters(),
			lr=LEARNING_RATE,
		)
	else:
		Optim = optim.SGD(
			Alex.parameters(),
			lr=LEARNING_RATE,
			momentum=MOMENTUM
		)

if __name__ == '__main__':
	if MODULE == 'LeNet5':#LeNet5
		print(Net5)
		if PRETRAIN == False:
			for epoch in range(EPOCH):
				sum_loss = 0.0
				for i, data in enumerate(train_loader):
					inputs, labels = data
					if DEVICE.type == 'gpu':
						inputs, labels = Variable(inputs.cuda(), Variable(labels).cuda())
					Optim.zero_grad()
					outputs = Net5(inputs)
					loss = Loss_Function(outputs, labels)
					loss.backward()
					Optim.step()

					sum_loss += loss.item()
					if i % 100 == 99:
						print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
						sum_loss = 0.0
			torch.save(Net5, PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(BATCH_SIZE) + '.pkl')
			print("Module Saved.")
		if PRETRAIN == True:
			if os.path.isfile(PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(BATCH_SIZE) + '.pkl'):
				Net5 = torch.load(PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(BATCH_SIZE) + '.pkl')
				print("Module Loaded.")
			else:
				print("Module Don't Exist, Start Train.")
				for epoch in range(EPOCH):
					sum_loss = 0.0
					for i, data in enumerate(train_loader):
						inputs, labels = data
						if DEVICE.type == 'gpu':
							inputs, labels = Variable(inputs.cuda(), Variable(labels).cuda())
						Optim.zero_grad()
						outputs = Net5(inputs)
						loss = Loss_Function(outputs, labels)
						loss.backward()
						Optim.step()

						sum_loss += loss.item()
						if i % 100 == 99:
							print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
							sum_loss = 0.0
				torch.save(Net5, PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(BATCH_SIZE) + '.pkl')
				print("Module Saved.")
				print('Retrain Module Loaded.')
		Net5.eval()  # 将模型变换为测试模式
		correct = 0
		total = 0

		for data_test in test_loader:
			images, labels = data_test
			if DEVICE.type == 'gpu':
				images, labels = Variable(images).cuda(), Variable(labels).cuda()
			output_test = Net5(images)
			_, predicted = torch.max(output_test, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()

		print("Test acc: {0}".format(correct.item() /
		                             len(test_dataset)))
	else:#AlexNet
		print(Alex)
		if PRETRAIN == False:
			for epoch in range(EPOCH):
				sum_loss = 0.0
				for i, data in enumerate(train_loader):
					inputs, labels = data
					if DEVICE.type == 'gpu':
						inputs, labels = Variable(inputs.cuda(), Variable(labels).cuda())
					Optim.zero_grad()
					outputs = Alex(inputs)
					loss = Loss_Function(outputs, labels)
					loss.backward()
					Optim.step()

					sum_loss += loss.item()
					if i % 100 == 99:
						print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
						sum_loss = 0.0
			torch.save(Alex, PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
				BATCH_SIZE) + '.pkl')
			print("Module Saved.")
		if PRETRAIN == True:
			if os.path.isfile(PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
					BATCH_SIZE) + '.pkl'):
				Net5 = torch.load(PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
					BATCH_SIZE) + '.pkl')
				print("Module Loaded.")
			else:
				print("Module Don't Exist, Start Train.")
				for epoch in range(EPOCH):
					sum_loss = 0.0
					for i, data in enumerate(train_loader):
						inputs, labels = data
						if DEVICE.type == 'gpu':
							inputs, labels = Variable(inputs.cuda(), Variable(labels).cuda())
						Optim.zero_grad()
						outputs = Alex(inputs)
						loss = Loss_Function(outputs, labels)
						loss.backward()
						Optim.step()

						sum_loss += loss.item()
						if i % 100 == 99:
							print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
							sum_loss = 0.0
				torch.save(Alex, PATH + MODULE + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
					BATCH_SIZE) + '.pkl')
				print("Module Saved.")
				print('Retrain Module Loaded.')
		Alex.eval()  # 将模型变换为测试模式
		correct = 0
		total = 0

		for data_test in test_loader:
			images, labels = data_test
			if DEVICE.type == 'gpu':
				images, labels = Variable(images).cuda(), Variable(labels).cuda()
			output_test = Alex(images)
			_, predicted = torch.max(output_test, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()

		print("Test acc: {0}".format(correct.item() /
		                             len(test_dataset)))
