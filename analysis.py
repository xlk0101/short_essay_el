import ujson
import config
import torch
import os

from collections import Counter
from utils.zh_wiki import zh2Hans
from utils import com_utils
from tqdm import tqdm

# init params
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


def analysis_train_data():
    print("start use the fasttext model to predict test data")
    if not os.path.exists(fileConfig.dir_analysis):
        os.mkdir(fileConfig.dir_analysis)
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_analysis + fileConfig.file_analysis_train_untind, 'w', encoding='utf-8')
    count = 1
    for line in tqdm(train_file, 'find unmatch'):
        jstr = ujson.loads(line)
        mention_data = jstr['mention_data']
        for mention in mention_data:
            mention_text = mention['mention']
            mention_id = mention['kb_id']
            kb_entity = kb_dict.get(mention_id)
            is_match = False
            if kb_entity is not None:
                kb_subject = kb_entity['subject']
                kb_alias = kb_entity['alias']
                if kb_subject == mention_text:
                    is_match = True
                if not is_match:
                    for alia in kb_alias:
                        if alia == mention_text:
                            is_match = True
                if not is_match:
                    out_file.write('-' * 20)
                    out_file.write('\n')
                    out_file.write("num:{}--text_id:{}--text:{}".format(count, jstr['text_id'], jstr['text']))
                    out_file.write('\n')
                    out_file.write("not match:")
                    out_file.write('\n')
                    out_file.write('*' * 20)
                    out_file.write('\n')
                    out_file.write('mention_original: {}'.format(ujson.dumps(mention, ensure_ascii=False)))
                    out_file.write('\n')
                    out_file.write(
                        "kb: {}".format('subject:{} alias:{}'.format(kb_entity['subject'], kb_entity['alias'])))
                    out_file.write('\n')
                    out_file.write('*' * 20)
                    out_file.write('\n')
                    count += 1
    train_file.close()
    out_file.close()
    print("success analysis train file find miss match entities")


def get_type_len(item):
    return len(item.split('-'))


def analysis_kb_type():
    kb_file = open(fileConfig.dir_data + fileConfig.file_kb_data, 'r', encoding='utf-8')
    type_dict = {}
    # type_list = []
    for line in tqdm(kb_file, desc='analysis kb type'):
        jstr = ujson.loads(line)
        type_str = com_utils.get_kb_type(jstr['type'])
        # for type_str in jstr['type']:
        if type_dict.get(type_str) is None:
            type_dict[type_str] = 1
        else:
            type_dict[type_str] += 1
        # if type_list.__contains__(type_str):
        #     continue
        # type_list.append(type_str)
    # type_list.sort(key=get_type_len, reverse=True)
    print("type len:{}".format(len(type_dict)))
    # print("type len:{}".format(len(type_list)))
    for dict_type in Counter(type_dict).most_common():
        print(dict_type)
    # for list_type in type_list:
    #     print(list_type)


def test():
    tensor = torch.LongTensor([0] * 30).to(torch.device('cuda'))
    print(tensor)


if __name__ == '__main__':
    # anlalysis_data_set()
    # anlalysis_miss_text()
    # test()
    # analysis_train_data()
    analysis_kb_type()
