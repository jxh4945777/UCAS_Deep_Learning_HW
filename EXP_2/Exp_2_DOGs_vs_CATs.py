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
DATA_SET = 'DOGs_vs_CATs' #数据集选择 MNIST, F-MNIST
#DOGs_vs_CATs
#AlexNet 1_epoch: 0.8841 20_epoch: 0.9242 30_epoch: 0.9292

BATCH_SIZE = 64 #Batch_Size
SHUFFLE = True #是否打乱数据集
LEARNING_RATE = 0.001 #学习率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#是否使用GPU
EPOCH = 30 #Epoch
PRETRAIN = False #是否加载预训练模型
MODULE = 'AlexNet' #使用的网络 LeNet5, AlexNet
PATH = './module/' #模型的保存路径
OPTIM = 'Adam' #模型的优化方式 Adam or SGD
MOMENTUM = 0.9 #SGD动态率


#数据处理_图像增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#数据集载入
dataset_train = datasets.ImageFolder('./dataset' + '/train', transform)
dataset_test = datasets.ImageFolder('./dataset' + '/val', transform)

train_loader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size=BATCH_SIZE, shuffle=True)

#猫狗数据可视化
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1, 2, 0)
print(labels)
cv2.imshow('DOGs_vs_CATs', img)
key_pressed = cv2.waitKey(0)

#网络实现
if MODULE == 'AlexNet':
	# AlexNet
	class AlexNet(nn.Module):
		def __init__(self):
			super(AlexNet, self).__init__()
			self.cov1 = nn.Sequential(  # INPUT(3, 128, 128) OUTPUT(32, 128, 128)
				nn.Conv2d(
					in_channels=3,
					out_channels=64,
					kernel_size=11,
					stride=4,
					padding=2,
				),
				nn.ReLU(),  # 激活函数#data(32, 128, 128)
				nn.MaxPool2d(kernel_size=3, stride=2) # 池化层 #data(32, 64, 64)
			)

			self.cov2 = nn.Sequential(  # INPUT(32, 64, 64) OUTPUT(64, 32, 32)
				nn.Conv2d(
					in_channels=64,
					out_channels=192,
					kernel_size=5,
					stride=1,
					padding=2
				),
				nn.ReLU(),  # 激活函数
				nn.MaxPool2d(kernel_size=3, stride = 2)  # 池化层
			)
			self.cov3 = nn.Sequential(  # INPUT(64, 7, 7) OUTPUT(256, 3, 3)
				nn.Conv2d(  # 卷积层
					in_channels=192,
					out_channels=384,
					kernel_size=3,
					stride=1,
					padding=1
				),
				nn.ReLU(),  # 激活函数
				nn.Conv2d(  # 卷积层
					in_channels=384,
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
				nn.MaxPool2d(kernel_size=3, stride = 2),  # 池化层
				nn.AdaptiveAvgPool2d((6, 6))
			)
			self.fc = nn.Sequential(  # 全连接层 #INPUT(256*6*6) OUTPUT(10)
				nn.Linear(256*6*6, 4096),
				nn.BatchNorm1d(4096),
				nn.ReLU(),
				nn.Dropout(0.5),
				nn.Linear(4096, 4096),
				nn.BatchNorm1d(4096),
				nn.ReLU(),
				nn.Dropout(0.5),
				nn.Linear(4096, 2),
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
	if not os.path.exists(PATH):
		os.makedirs(PATH)

	print(Alex)
	if PRETRAIN == False:
		for epoch in range(EPOCH):
			sum_loss = 0.0
			for i, data in enumerate(train_loader):
				inputs, labels = data
				if DEVICE.type == 'cuda':
					inputs, labels = Variable(inputs.cuda()), Variable((labels).cuda())
				Optim.zero_grad()
				outputs = Alex(inputs)
				loss = Loss_Function(outputs, labels)
				loss.backward()
				Optim.step()

				sum_loss += loss.item()
				if i % 100 == 99:
					print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
					sum_loss = 0.0
		torch.save(Alex, PATH + MODULE + '_' + DATA_SET + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
			BATCH_SIZE) + '.pkl')
		print("Module Saved.")
	if PRETRAIN == True:
		if os.path.isfile(PATH + MODULE + '_' + DATA_SET + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
				BATCH_SIZE) + '.pkl'):
			Alex = torch.load(PATH + MODULE + '_' + DATA_SET + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
				BATCH_SIZE) + '.pkl')
			print("Module Loaded.")
		else:
			print("Module Isn't Exist, Start Train.")
			for epoch in range(EPOCH):
				sum_loss = 0.0
				for i, data in enumerate(train_loader):
					inputs, labels = data
					if DEVICE.type == 'cuda':
						inputs, labels = Variable(inputs.cuda()), Variable((labels).cuda())
					Optim.zero_grad()
					outputs = Alex(inputs)
					loss = Loss_Function(outputs, labels)
					loss.backward()
					Optim.step()

					sum_loss += loss.item()
					if i % 100 == 99:
						print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
						sum_loss = 0.0
			torch.save(Alex, PATH + MODULE + '_' + DATA_SET + '_' + OPTIM + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(
				BATCH_SIZE) + '.pkl')
			print("Module Saved.")
			print('Retrain Module Loaded.')
	Alex.eval()  # 将模型变换为测试模式
	correct = 0
	total = 0

	for data_test in test_loader:
		images, labels = data_test
		if DEVICE.type == 'cuda':
			images, labels = Variable(images).cuda(), Variable(labels).cuda()
		output_test = Alex(images)
		_, predicted = torch.max(output_test, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	print("Test acc: {0}".format(correct.item() /
	                             len(dataset_test)))
