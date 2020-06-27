import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os.path
import numpy as np
import torchnet as tnt

#参数
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
BATCH_SIZE = 16
MAX_GEN_LEN = 125
EPOCH = 20
start_words = "湖光秋月两相和"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#是否使用GPU


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
		self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
		self.linear = nn.Linear(self.hidden_dim, vocab_size)

	def forward(self, input, hidden=None):
		embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
		batch_size, seq_len = input.size()
		if hidden is None:
			h_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
			c_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
		else:
			h_0, c_0 = hidden
		output, hidden = self.lstm(embeds, (h_0, c_0))
		output = self.linear(output)
		output = output.reshape(batch_size * seq_len, -1)
		return output, hidden

# 定义模型
model = PoetryModel(len(word2ix), embedding_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
loss_meter = tnt.meter.AverageValueMeter()

print("Start Train.....")

for epoch in range(EPOCH):
	loss_meter.reset()
	sum_loss = 0.0
	for i, data_ in enumerate(dataloader):
		# 训练
		data_ = data_.long().transpose(1, 0).contiguous()
		data_ = data_.to(DEVICE)
		optimizer.zero_grad()
		input_, target = data_[:-1, :], data_[1:, :]
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

results = list(start_words)
start_words_len = len(start_words)
# 第一个词语是<START>
input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
hidden = None

model.eval()
with torch.no_grad():
	for i in range(MAX_GEN_LEN):
		output, hidden = model(input, hidden)
		# 如果在给定的句首中，input为句首中的下一个字
		if i < start_words_len:
			w = results[i]
			input = input.data.new([word2ix[w]]).view(1, 1)
			if DEVICE.type == 'cuda':
				input = Variable(input).cuda()
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
print(results)






