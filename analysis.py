import ujson
import config
import torch

from collections import Counter
from utils.zh_wiki import zh2Hans
from utils import zh_wiki

fileConfig = config.FileConfig()


def anlalysis_data_set():
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, 'r')
    dev_file = open(fileConfig.dir_data + fileConfig.file_dev_data)
    text_list = []
    max_length = 0
    text_len_dict = {}
    for line in train_file:
        jstr = ujson.loads(line)
        text = jstr['text'].strip()
        text_len = len(text)
        if not text_len_dict.get(text_len):
            text_len_dict[text_len] = 1
        else:
            text_len_dict[text_len] += 1
        if text_len > max_length:
            max_length = text_len
        text_list.append(text)
    for line in dev_file:
        jstr = ujson.loads(line)
        text = jstr['text'].strip()
        text_len = len(text)
        if not text_len_dict.get(text_len):
            text_len_dict[text_len] = 1
        else:
            text_len_dict[text_len] += 1
        if text_len > max_length:
            max_length = text_len
        text_list.append(text)
    data_len = len(text_list)
    print("the data set length is {}".format(data_len))
    print("the max text length is {}".format(max_length))
    print(Counter(text_len_dict).most_common())


def anlalysis_miss_text():
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, 'r')
    dev_file = open(fileConfig.dir_data + fileConfig.file_dev_data, 'r')
    bert_vocab_file = open(fileConfig.dir_bert + fileConfig.file_bert_vocab, 'r')
    # out_vocab = open(fileConfig.dir_analysis + fileConfig.file_analysis_vocab, 'w')
    # out_unfind = open(fileConfig.dir_analysis + fileConfig.file_analysis_unfind, 'w')
    # create bert vocab
    bert_dict = {}
    for line in bert_vocab_file:
        text = line.strip('\n')
        if not bert_dict.get(text):
            bert_dict[text] = 1
        else:
            bert_dict[text] += 1
    print("success create bert vocab")

    text_dict = {}
    for line in train_file:
        jstr = ujson.loads(line)
        text = jstr['text'].strip()
        text_list = list(text)
        for text in text_list:
            if not text_dict.get(text):
                text_dict[text] = 1
            else:
                text_dict[text] += 1
    for line in dev_file:
        jstr = ujson.loads(line)
        text = jstr['text'].strip()
        text_list = list(text)
        for text in text_list:
            if not text_dict.get(text):
                text_dict[text] = 1
            else:
                text_dict[text] += 1
    # for item in Counter(text_dict).most_common():
    #     text = '{} - {}'.format(item[0], item[1])
    #     out_vocab.write(text + '\n')
    # print("success write train/dev vocab")

    # filter text which not in bert vocab
    unfind_dict = {}
    for item in text_dict:
        if bert_dict.get(item) is None:
            if not unfind_dict.get(item):
                unfind_dict[item] = 1
            else:
                unfind_dict[item] += 1
    # for item in Counter(unfind_dict).most_common():
    #     text = '{} - {}'.format(item[0], item[1])
    #     out_unfind.write(text + '\n')
    # print("success write unfind vocab file")

    # convert fan to jian
    print("1. unfind dict init len {}".format(len(unfind_dict)))
    count = 0
    for item in unfind_dict:
        if zh2Hans.get(item) is not None:
            count += 1
    print("2. unfind dict deal fan 2 jian len {}".format(len(unfind_dict) - count))


def test():
    tensor = torch.LongTensor([0]*30).to(torch.device('cuda'))
    print(tensor)

if __name__ == '__main__':
    # anlalysis_data_set()
    # anlalysis_miss_text()
    test()
