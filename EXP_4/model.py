import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class CNN_LSTM(nn.Module):

	def __init__(self, index2class, args):
		super(CNN_LSTM, self).__init__()
		self.args = args

		#CNN部分
		EMBED_NUM = args.VOCAB_SIZE
		EMBED_DIM = args.EMBEDDING_DIM
		CLASS_NUM = args.NUM_CLASSES
		Ci = 1
		Co = args.KERNEL_NUM
		Ks = args.KERNEL_SIZES
		self.HIDDEN_SIZE = args.HIDDEN_DIM

		self.embed = nn.Embedding(EMBED_NUM, EMBED_DIM)

		self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, EMBED_DIM)) for K in Ks])
		'''
		self.conv13 = nn.Conv2d(Ci, Co, (3, D))
		self.conv14 = nn.Conv2d(Ci, Co, (4, D))
		self.conv15 = nn.Conv2d(Ci, Co, (5, D))
		'''

		self.dropout = nn.Dropout(args.DROPOUT)
		self.fc1 = nn.Linear(len(Ks)*Co, self.HIDDEN_SIZE)

		self.fc_sub = nn.Sequential(
			nn.Linear(2*self.HIDDEN_SIZE, CLASS_NUM)
		)
		self.fc2 = nn.Linear(self.HIDDEN_SIZE, CLASS_NUM)
		self.fc3 = nn.Linear(self.HIDDEN_SIZE, CLASS_NUM)

		# LSTM部分
		self.lstm = nn.LSTM(EMBED_DIM, args.HIDDEN_DIM, num_layers = args.NUM_LAYERS, batch_first=True, bidirectional=True)

	def self_attention(self, x):
		hidden_size = self.HIDDEN_SIZE
		Q = x
		K = x
		V = x
		attetnion_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1))/math.sqrt(hidden_size), -1)
		A = torch.bmm(attetnion_weight, V)
		A = A.permute(0, 2, 1)
		x = F.max_pool1d(A, A.size()[2]).squeeze(-1)
		return x

	def CNN_process(self, x):
		if self.args.EMBEDDING_STATIC == True:
			x = Variable(x)
		x = x.unsqueeze(1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x, 1)
		x = self.dropout(x)
		x = self.fc1(x)
		return x

	def LSTM_process(self, x):
		output, (hn, cn) = self.lstm(x)
		x = self.self_attention(x)
		return x

	def forward(self, x):
		x = self.embed(x)
		x_CNN = self.CNN_process(x) #CNN
		x_LSTM = self.LSTM_process(x) #LSTM
		x_SUB = torch.cat((x_CNN, x_LSTM), -1)
		logit = self.fc_sub(x_SUB)
		# logit = self.fc3(x_LSTM)
		# logit = self.fc2(x_CNN)
		return logit


