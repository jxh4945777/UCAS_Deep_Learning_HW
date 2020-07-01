import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_LSTM(nn.Module):

	def __init__(self, args):
		super(CNN_LSTM, self).__init__()
		self.args = args
		V = args.embed_num
		D = args.embed_dim
		C = args.class_num
		Ci = 1
		Co = args.kernel_num
		Ks = args.kernel_sizes

		self.embed = nn.Embedding(V, D)
		# self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
		self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
		'''
		self.conv13 = nn.Conv2d(Ci, Co, (3, D))
		self.conv14 = nn.Conv2d(Ci, Co, (4, D))
		self.conv15 = nn.Conv2d(Ci, Co, (5, D))
		'''
		self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(len(Ks)*Co, C)