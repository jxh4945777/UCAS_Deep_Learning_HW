import os
import argparse
import sys
import torch
import torch.nn.functional as F
import torchtext.data as data
from torchtext.vocab import Vectors
import torch.optim as optim
import data_preprocess
from model import CNN_LSTM
from sklearn import metrics

torch.cuda.set_device(0)
# torch.backends.cudnn.enabled = False

#参数

DEVICE_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#是否使用GPU

# parser = argparse.ArgumentParser(description='Movie_Review_Classification')
# parser.add_argument('-LEARNING_RATE', type=float, default=0.001, help='')
# parser.add_argument('-L2',type=float,default=0,help='')
# parser.add_argument('-EPOCH', type=int, default=10, help='')
# parser.add_argument('-BATCH_SIZE', type=int, default=32, help='')
# parser.add_argument('-EMBEDDING_DIM', type=int, default=300, help='')
# parser.add_argument('-HIDDEN_DIM', type=int, default=300, help='')
# parser.add_argument('-KERNEL_NUM', type=int, default=100, help='')
# parser.add_argument('-NUM_LAYERS', type=int, default=3, help='')
# parser.add_argument('-NUM_CLASSES', type=int, default=2, help='')
# parser.add_argument('-DROPOUT', type=float, default=0.3, help='')
# parser.add_argument('-SEQUENCE_LENGTH', type=int, default=50, help='')
# parser.add_argument('-OPTIM', type=str, default="Adam", help='')
# parser.add_argument('-EARLY_STOPPING', type=int, default=400, help='')
# parser.add_argument('-KERNEL_SIZES', type=str, default='3,4,5', help='')
# parser.add_argument('-MOMENTUM', type=float, default=0.9, help='')
#
# parser.add_argument('-DEVICE', type=str, default=DEVICE_.type, help='')
# parser.add_argument('-MODULE_PATH', type=str, default="./module/", help='')
# parser.add_argument('-PRETRAIN', type=bool, default=False, help='')
# parser.add_argument('-PRETRAIN_PATH', type=str, default="./module/model.pkl", help='')
# parser.add_argument('-ALREADY_WORD_CUT', type=bool, default=True, help='')
# parser.add_argument('-PRE_EMBEDDING', type=bool, default=True, help='')
# parser.add_argument('-EMBEDDING_PATH', type=str, default="./embedding/sgns.sogou.char", help='')
# parser.add_argument('-EMBEDDING_STATIC', type=bool, default=False, help='')
# parser.add_argument('-SAVE_BEST', type=bool, default=True, help='')
# parser.add_argument('-SAVE_DIR', type=str, default="./snapshot", help='')
# parser.add_argument('-TRAIN_DATA', type=str, default="train_new.txt", help='')
# parser.add_argument('-VAL_DATA', type=str, default="validation_new.txt", help='')
# parser.add_argument('-TEST_DATA', type=str, default="test_new.txt", help='')
#
#
# parser.add_argument('-LOG_INTERVAL', type=int, default=1,help='')
# parser.add_argument('-TEST_INTERVAL', type=int, default=100,help='')
#
# args = parser.parse_args()

parser = argparse.ArgumentParser(description='Movie_Review_Classification')
parser.add_argument('-LEARNING_RATE', type=float, default=0.001, help='')
parser.add_argument('-L2',type=float,default=0,help='')
parser.add_argument('-EPOCH', type=int, default=20, help='')
parser.add_argument('-BATCH_SIZE', type=int, default=32, help='')
parser.add_argument('-EMBEDDING_DIM', type=int, default=50, help='')
parser.add_argument('-HIDDEN_DIM', type=int, default=50, help='')
parser.add_argument('-KERNEL_NUM', type=int, default=256, help='')
parser.add_argument('-NUM_LAYERS', type=int, default=3, help='')
parser.add_argument('-NUM_CLASSES', type=int, default=2, help='')
parser.add_argument('-DROPOUT', type=float, default=0.5, help='')
parser.add_argument('-SEQUENCE_LENGTH', type=int, default=50, help='')
parser.add_argument('-OPTIM', type=str, default="Adam", help='')
parser.add_argument('-EARLY_STOPPING', type=int, default=800, help='')
parser.add_argument('-KERNEL_SIZES', type=str, default='3,4,5', help='')
parser.add_argument('-MOMENTUM', type=float, default=0.9, help='')

parser.add_argument('-DEVICE', type=str, default=DEVICE_.type, help='')
parser.add_argument('-MODULE_PATH', type=str, default="./module/", help='')
parser.add_argument('-PRETRAIN', type=bool, default=False, help='')
parser.add_argument('-PRETRAIN_PATH', type=str, default="./module/model.pkl", help='')
parser.add_argument('-ALREADY_WORD_CUT', type=bool, default=True, help='')
parser.add_argument('-PRE_EMBEDDING', type=bool, default=True, help='')
parser.add_argument('-EMBEDDING_PATH', type=str, default="./embedding/embedding.50", help='')
parser.add_argument('-EMBEDDING_STATIC', type=bool, default=True, help='')
parser.add_argument('-SAVE_BEST', type=bool, default=True, help='')
parser.add_argument('-SAVE_DIR', type=str, default="./snapshot", help='')
parser.add_argument('-TRAIN_DATA', type=str, default="train_new.txt", help='')
parser.add_argument('-VAL_DATA', type=str, default="validation_new.txt", help='')
parser.add_argument('-TEST_DATA', type=str, default="test_new.txt", help='')


parser.add_argument('-LOG_INTERVAL', type=int, default=1,help='')
parser.add_argument('-TEST_INTERVAL', type=int, default=100,help='')

args = parser.parse_args()



def load_word_vectors(embedding_path):#加载词向量
	vectors = Vectors(name=embedding_path)
	return vectors

def load_dataset(text_field, label_field, args, **kwargs):
	train_dataset, dev_dataset, test_dataset = data_preprocess.get_dataset('dataset', text_field, label_field, args.ALREADY_WORD_CUT, args)
	if args.PRE_EMBEDDING:#有预训练的词向量
		print("loading_word_embedding.....")
		vectors = load_word_vectors(args.EMBEDDING_PATH)
		text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)

	else:
		print("no_pre-trained_word_embedding.....")
		text_field.build_vocab(train_dataset, dev_dataset)

	label_field.build_vocab(train_dataset, dev_dataset)

	train_iter, dev_iter, test_iter = data.Iterator.splits(
		(train_dataset, dev_dataset,test_dataset),
		#显存不一定够用
		batch_sizes=(args.BATCH_SIZE, args.BATCH_SIZE, args.BATCH_SIZE),
		sort_key=lambda x: len(x.text),#指定分批次的方法
		**kwargs, sort_within_batch = False, sort = False)#切分batch
	return train_iter, dev_iter, test_iter

def clip_gradient(model,clip_value):
	params = list(filter(lambda p:p.grad is not None,model.parameters()))
	for p in params:
		p.grad.data.clamp_(-clip_value,clip_value)

def save(model, save_dir, save_prefix, epoch, steps):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, save_prefix)
	save_path = '{}_epoch_{}_steps_{}.pt'.format(save_prefix,epoch, steps)
	torch.save(model.state_dict(), save_path)
	return save_path

def train_model(model, train_iter, dev_iter, args):
	model.train()
	if args.DEVICE == "cuda":
		model.cuda()
	if args.OPTIM == 'sgd':
		optimizer = optim.SGD(model.parameters(),lr=args.LEARNING_RATE,weight_decay=args.L2)
	elif args.OPTIM == 'momentum':
		optimizer = optim.SGD(model.parameters(), lr=args.LEARNING_RATE, momentum=args.MOMENTUM, weight_decay=args.L2,nesterov=True)
	else:
		optimizer = optim.Adam(model.parameters(),lr=args.LEARNING_RATE,betas=(0.9,0.999),eps=1e-8,weight_decay=args.L2)
	best_acc = 0
	last_step = 0
	saved_model_name = ''
	finish_train = False
	for epoch in range(1, args.EPOCH + 1):
		if finish_train == True:
			break
		steps = 0
		last_epoch = 0
		max_step = 0
		for batch in train_iter:
			short_text, target = batch.text, batch.cluster_name
			with torch.no_grad():
				short_text.t_(), target.sub_(1)
			if args.DEVICE == "cuda":
				short_text, target = short_text.cuda(), target.cuda()
			optimizer.zero_grad()
			logits = model(short_text)
			loss = F.cross_entropy(logits, target)
			loss.backward()
			clip_gradient(model, 1) #梯度裁剪
			optimizer.step()
			if steps > max_step:
				max_step = steps
			if steps % args.LOG_INTERVAL == 0:
				corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
				corrects_ = corrects.tolist()
				train_acc = 100.00 * corrects_  / batch.batch_size
				sys.stdout.write('\rTRAIN: Epoch[{}] - Step[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,steps,loss.item(),
			                                                                                            train_acc,corrects,batch.batch_size))
			steps += 1

			if steps % args.TEST_INTERVAL == 0:
				dev_acc = eval_model(model, dev_iter, args)
				if dev_acc > best_acc:
					best_acc = dev_acc
					last_step = steps
					last_epoch = epoch
					if args.SAVE_BEST:
						print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
						saved_model_name = save(model, args.SAVE_DIR, 'best',epoch, steps)
				else:
					if (epoch - last_epoch) + (steps -last_step) >= args.EARLY_STOPPING:
						print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.EARLY_STOPPING, best_acc))
						finish_train = True
						break
				model.train()
	return  saved_model_name

def eval_model(model,data_iter,args):
	size, corrects, avg_loss = 0, 0, 0
	model.eval()
	for batch in data_iter:
		short_text, target = batch.text, batch.cluster_name
		with torch.no_grad():  # 不需要梯度修改的内容
			short_text.t_(), target.sub_(1) # _t()代表转置
		if args.DEVICE == "cuda":
			short_text, target = short_text.cuda(), target.cuda()
		logits =  model(short_text)
		loss = F.cross_entropy(logits, target)
		avg_loss += loss.item()
		corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()#这个是评价 ACC 如果做F1，P，R；也在此基础上实现
		# print(s)

		# print(str(torch.max(logits, 1)[1].view(target.size()).data))
		size += batch.batch_size
	avg_loss /= size
	corrects_ = corrects.tolist()
	accuracy = 100.00 * corrects_ / size
	print('\nEvaluation: loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,accuracy,corrects,size))
	return accuracy

def test_model(model,data_iter,args,index2class):
	size, corrects, avg_loss = 0, 0, 0
	model.eval()
	A = []
	B = []
	for batch in data_iter:
		short_text, target = batch.text, batch.cluster_name
		with torch.no_grad():  # 不需要梯度修改的内容
			short_text.t_(), target.sub_(1)  # _t()代表转置
		if args.DEVICE == "cuda":
			short_text, target = short_text.cuda(), target.cuda()
		logits =  model(short_text)

		loss = F.cross_entropy(logits, target)
		avg_loss += loss.item()
		corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()#这个是评价 ACC 如果做F1，P，R；也在此基础上实现
		A.append(torch.max(logits, 1)[1].view(target.size()).tolist()[:])
		B.append(target.data.tolist()[:])
		s = torch.max(logits, 1)[1].view(target.size()).data.cpu().numpy()
		# print(str(torch.max(logits, 1)[1].view(target.size()).data))
		size += batch.batch_size
	avg_loss /= size
	corrects_ = corrects.tolist()
	accuracy = 100.0 * corrects_ / size
	# mylog = open('my_log'+str(args.EMBEDDING_DIM) + '.txt', mode='a', encoding='utf-8')

	s = torch.max(logits, 1)[1].view(target.size()).data.cpu().numpy()
	output_file_entity = open('s_' + str(args.EMBEDDING_DIM) + '.txt', 'a+', encoding='utf-8')
	output_file_entity.write(str(s))
	output_file_entity.close()

	print('\nEvaluation: loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,accuracy,corrects,size))
	# print(index2class, file = mylog)
	y_test =[]
	y_predict = []
	for i_batch in A:
		for i_label in i_batch:
			y_predict.append(i_label)
	for j_batch in B:
		for j_label in j_batch:
			y_test.append(j_label)

	print('准确率:', metrics.accuracy_score(y_test, y_predict))  # 预测准确率输出

	print('精确率:', metrics.precision_score(y_test, y_predict, average='macro'))  # 预测宏平均精确率输出

	print('召回率:', metrics.recall_score(y_test, y_predict, average='macro'))  # 预测宏平均召回率输出

	print('F1-score:', metrics.f1_score(y_test, y_predict, average='macro'))  # 预测宏平均f1-score输出

	print('分类报告:\n', metrics.classification_report(y_test, y_predict))  # 分类报告输出
	return accuracy

if __name__ == "__main__":

	print('Loading data...')

	# 数据载入的核心部分
	text_field = data.Field(lower=True)#具体文本
	label_field = data.Field(sequential=False)#每个文本的标签


	train_iter, dev_iter, test_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)#数据载入
	args.VOCAB_SIZE = len(text_field.vocab)#文章中的表达词汇数目
	args.OUTPUT_SIZE = args.HIDDEN_DIM
	args.BEST_MODEL = "best_" + str(args.EMBEDDING_DIM) + ".pt"
	args.KERNEL_SIZES = [int(k) for k in args.KERNEL_SIZES.split(',')]

	#创建vocab字典
	index2class = label_field.vocab.itos[1:]
	index2class = list(map(int, index2class))

	print('Parameters:')

	for attr, value in sorted(args.__dict__.items()):
		if attr in {'vectors'}:
			continue
		print('\t{}={}'.format(attr.upper(), value))

	model = CNN_LSTM(index2class, args)
	model_name = train_model(model, train_iter, dev_iter, args)
	print('\n'+model_name)

	if args.DEVICE == 'cuda':
		model.cuda()

	model.load_state_dict(torch.load(model_name, map_location='cpu'))
	test_acc = test_model(model,test_iter,args, index2class)
	print(f'Test Acc: {test_acc:.2f}%')


