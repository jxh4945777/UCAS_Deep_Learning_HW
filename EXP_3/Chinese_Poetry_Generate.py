import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import os.path
import numpy as np
import torchnet as tnt

#参数
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100
HIDDEN_DIM = 1024
BATCH_SIZE = 64
MAX_GEN_LEN = 125
EPOCH = 20
START_WORDS = "湖光秋月两相和"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#是否使用GPU
NUM_LAYERS = 2



# class PoemDataSet(Dataset):
# 	def __init__(self,poem_path,seq_len):
# 		self.seq_len = seq_len
# 		self.poem_path = poem_path
# 		self.poem_data, self.ix2word, self.word2ix = self.get_raw_data()
# 		self.no_space_data = self.filter_space()
#
# 	def __getitem__(self, idx:int):
# 		txt = self.no_space_data[idx*self.seq_len : (idx+1)*self.seq_len]
# 		label = self.no_space_data[idx*self.seq_len + 1 : (idx+1)*self.seq_len + 1] # 将窗口向后移动一个字符就是标签
# 		txt = torch.from_numpy(np.array(txt)).long()
# 		label = torch.from_numpy(np.array(label)).long()
# 		return txt,label
#
# 	def __len__(self):
# 		return int(len(self.no_space_data) / self.seq_len)
#
# 	def filter_space(self): # 将空格的数据给过滤掉，并将原始数据平整到一维
# 		t_data = torch.from_numpy(self.poem_data).view(-1)
# 		flat_data = t_data.numpy()
# 		no_space_data = []
# 		for i in flat_data:
# 			if (i != 8292):
# 				no_space_data.append(i)
# 		return no_space_data
# 	def get_raw_data(self):
# 		#         datas = np.load(self.poem_path,allow_pickle=True)  #numpy 1.16.2  以上引入了allow_pickle
# 		datas = np.load(self.poem_path, allow_pickle=True)
# 		data = datas['data']
# 		ix2word = datas['ix2word'].item()
# 		word2ix = datas['word2ix'].item()
# 		return data, ix2word, word2ix

# poem_ds = PoemDataSet("./dataset/tang.npz", 125)
# ix2word = poem_ds.ix2word
# word2ix = poem_ds.word2ix
# dataloader =  DataLoader(poem_ds, batch_size=BATCH_SIZE, shuffle=True)




datas = np.load("./dataset/tang.npz", allow_pickle = True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
data = torch.from_numpy(data)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

VOCAB_SIZE = len(word2ix)

class PoetryModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(PoetryModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=NUM_LAYERS, batch_first=True, bidirectional= False)
		self.linear = nn.Linear(self.hidden_dim, vocab_size)

	def forward(self, input, hidden=None):
		embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
		batch_size, seq_len = input.size()
		if hidden is None:
			h_0 = input.data.new(NUM_LAYERS, batch_size, self.hidden_dim).fill_(0).float()
			c_0 = input.data.new(NUM_LAYERS, batch_size, self.hidden_dim).fill_(0).float()
		else:
			h_0, c_0 = hidden
		output, hidden = self.lstm(embeds, (h_0, c_0))
		output = self.linear(output)
		output = output.reshape(batch_size * seq_len, -1)
		return output, hidden

# 定义模型
model = PoetryModel(VOCAB_SIZE, embedding_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM).to(DEVICE)
model.train()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
loss_meter = tnt.meter.AverageValueMeter()

print("Start Train.....")

for epoch in range(EPOCH):
	loss_meter.reset()
	sum_loss = 0.0
	for i, data_ in enumerate(dataloader, 0):
		# 训练
		# data_ = data_.long().transpose(1, 0).contiguous()
		# data_ = data_.to(DEVICE)
		optimizer.zero_grad()
		# input_, target = data_[:-1, :], data_[1:, :]
		input_, target = data_[0].to(DEVICE), data_[1].to(DEVICE)
		now_str = ''
		input__ = input_.view(-1)

		# for k in range(input_.view(-1).shape[0]):
		# 	now_str += ix2word[int(input_.view(-1)[k])]
		# print(now_str)

		if DEVICE.type == 'cuda':
			input_, target = Variable(input_.cuda()), Variable((target).cuda())
		output, _ = model(input_)
		loss = criterion(output, target.view(-1))
		loss.backward()
		optimizer.step()
		loss_meter.add(loss.item())
		sum_loss += loss.item()
		if i % 100 == 99:
			print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
			sum_loss = 0.0

results = list(START_WORDS)
start_words_len = len(START_WORDS)
# 第一个词语是<START>
input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
# hidden = torch.zeros((2, NUM_LAYERS*1, 1, HIDDEN_DIM), dtype=torch.float)
hidden = None
if DEVICE.type == 'cuda':
	input = Variable(input).cuda()
	# hidden = Variable(hidden).cuda()
model.eval()

with torch.no_grad():
	for i in range(MAX_GEN_LEN):
		output, hidden = model(input, hidden)
		# 如果在给定的句首中，input为句首中的下一个字
		if i < start_words_len:
			w = results[i]
			input = input.data.new([word2ix[w]]).view(1, 1)
			# if DEVICE.type == 'cuda':
			# 	input = Variable(input).cuda()
		# 否则将output作为下一个input进行
		else:

			top_index = output.data[0].topk(1)[1][0].item()
			w = ix2word[top_index]
			results.append(w)
			input = input.data.new([top_index]).view(1, 1)
			if DEVICE.type == 'cuda':
				input = Variable(input).cuda()
		if w == '<EOP>':
			del results[-1]
			break
final_poem = ""
for zi in results:
	final_poem += str(zi)
print(final_poem)






