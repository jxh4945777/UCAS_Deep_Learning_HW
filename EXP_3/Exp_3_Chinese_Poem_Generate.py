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
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCH = 50
START_WORDS = "湖光秋月两相和"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#是否使用GPU
NUM_LAYERS = 3
TYPE = "Seven" #Seven 34    Five 26
TYPE_LEN = {"Seven": 34, "Five": 26}
MODULE_PATH = "./module/"
PRETRAIN = True

#数据加载
def data_init():
	ix2word = {}
	word2ix = {}
	for index_line in open("./dataset/char_index.txt", "r", encoding = "utf-8"):
		split_line = index_line.replace("\n","").split(" ")
		ix = int(split_line[0])
		word = split_line[1]
		ix2word[ix] = word
		word2ix[word] = ix
	print("Init_Index")

	data_ = []
	for poem_line_ in open("./dataset/" + TYPE + "_Poem_Data.txt", "r", encoding = "utf-8"):
		poem_line = poem_line_.replace("\n", "")
		int_poem_line = []
		for each_char in poem_line:
			int_poem_line.append(word2ix[each_char])
		data_.append(int_poem_line)
	data = torch.tensor(data_, dtype=torch.int64)
	print("Init_Data")
	return data, ix2word, word2ix

#模型架构
class PoetryModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(PoetryModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers = NUM_LAYERS)
		self.linear = nn.Linear(self.hidden_dim, vocab_size)

	def forward(self, input, hidden=None):
		embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
		seq_len, batch_size = input.size()
		if hidden is None:
			h_0 = input.data.new(NUM_LAYERS, batch_size, self.hidden_dim).fill_(0).float()
			c_0 = input.data.new(NUM_LAYERS, batch_size, self.hidden_dim).fill_(0).float()
		else:
			h_0, c_0 = hidden
		output, hidden = self.lstm(embeds, (h_0, c_0))
		output = self.linear(output)
		output = output.view(batch_size * seq_len, -1)
		return output, hidden


def poem_train(model, dataloader, optimizer, criterion, loss_meter, ix2word, word2ix):
	print("Start Train.....")
	model.train()
	for epoch in range(EPOCH):
		model.train()
		loss_meter.reset()
		sum_loss = 0.0
		for i, data_ in enumerate(dataloader):
			# 训练
			data_ = data_.long().transpose(1, 0).contiguous()
			data_ = data_.to(DEVICE)
			optimizer.zero_grad()
			input_, target = data_[:-1, :].to(DEVICE), data_[1:, :].to(DEVICE)

			output, _ = model(input_)
			loss = criterion(output, target.view(-1))
			loss.backward()
			optimizer.step()
			loss_meter.add(loss.item())
			sum_loss += loss.item()
			if i % 100 == 99:
				print('Epoch: %d, Step: %d loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
				sum_loss = 0.0
		results = poem_generate(model, START_WORDS, ix2word, word2ix)
		test_poem = 'Test: '
		for zi in results:
			test_poem += str(zi)
		print(test_poem)

def poem_generate(model, input_words_, ix2word, word2ix):
	# input_words = "<" + input_words_
	input_words_len = len(input_words_)
	results = list(input_words_)
	input = torch.Tensor([word2ix['<']]).view(1, 1).long()
	hidden = None
	if DEVICE.type == 'cuda':
		input = Variable(input).cuda()
	# hidden = Variable(hidden).cuda()
	model.eval()
	with torch.no_grad():
		for i in range(TYPE_LEN[TYPE]-2):
			output, hidden = model(input, hidden)
			# 如果在给定的句首中，input为句首中的下一个字
			if i < input_words_len:#这里看看有没有BUG
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
	return results

if __name__ == '__main__':
	data, ix2word, word2ix = data_init()
	dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle = True)
	VOCAB_SIZE = len(word2ix)

	# 定义模型
	model = PoetryModel(VOCAB_SIZE, embedding_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM).to(DEVICE)
	optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
	criterion = nn.CrossEntropyLoss()
	loss_meter = tnt.meter.AverageValueMeter()
	if PRETRAIN == False:
		poem_train(model, dataloader, optimizer, criterion, loss_meter, ix2word, word2ix)
		if os.path.exists(MODULE_PATH) == False:
			os.mkdir(MODULE_PATH)
		torch.save(model.state_dict(), MODULE_PATH + TYPE + "_module.pth")

	model.load_state_dict(torch.load(MODULE_PATH + TYPE + "_module.pth"))  # 模型加载

	results = poem_generate(model, START_WORDS, ix2word, word2ix)
	final_poem = 'Final: '
	for zi in results:
		final_poem += str(zi)
	print(final_poem)