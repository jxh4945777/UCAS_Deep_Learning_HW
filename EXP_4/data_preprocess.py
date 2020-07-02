import re
from torchtext import data
import jieba
import logging
jieba.setLogLevel(logging.INFO)
from torchtext import data
import torchtext
import torch
import logging
LOGGER = logging.getLogger("导入数据")

def _clean_data(sent, sw=None, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。]", " ", sent)
        sent = re.sub('！{2,}', '！', sent)
        sent = re.sub('？{2,}', '！', sent)
        sent = re.sub('。{2,}', '。', sent)
        sent = re.sub('，{2,}', '，', sent)
        sent = re.sub('\s{2,}', ' ', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    if sw is not None:
        sent = "".join([word for word in sent if word not in sw])

    return sent

def word_cut_already(text):
    text = _clean_data(text)
    text_list = text.split(" ")
    return text_list

def word_cut(text):
    text = _clean_data(text)
    return [word for word in jieba.cut(text) if word.strip()]

def get_stop_words():
    file_object = open('./dataset/ChineseStopWords.txt', encoding='UTF-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

def get_dataset(path, text_field, label_field, already_cut, args):#该文件设置读取的内容
    stop_words = get_stop_words()  # 加载停用词表
    if already_cut == True:
        text_field.tokenize = word_cut_already
    else:
        text_field.tokenize = word_cut#是否用jieba分词 (word_cut, word_cut_already)
    text_field.stop_words = stop_words#去除停用词
    train, dev,test = data.TabularDataset.splits(
        path=path, format='csv', skip_header=True,
        train=args.TRAIN_DATA, validation=args.VAL_DATA, test=args.TEST_DATA,
        fields=[#根据header判定
            ('index', None),
            ('cluster_name', label_field),
            ('text', text_field)
        ]
    )
    return train, dev,test

def process_dataset(file_name):
    new_file_name = file_name.replace(".txt","") + "_new.txt"
    new_file = open(new_file_name,"w+",encoding="utf-8")
    new_file.write("id,cluster_name,text\n")
    index = 1
    for line in open(file_name,"r",encoding="utf-8"):
        id = str(line[0])
        text = str(line[2:])
        if len(text) == 0:
            break
        new_file.write(str(index) + "," + id + "," + text)
        index += 1

#首先处理下课程给的数据集
process_dataset("./dataset/train.txt")
process_dataset("./dataset/validation.txt")
process_dataset("./dataset/test.txt")


