"""
use fasttext to train word vec and find mention
"""
import sys
import os
import gensim
import numpy as np
import config
import fastText
import random
import ujson
import ast
import pandas as pd
import jieba_fast as jieba
from tqdm import tqdm
from utils import data_utils, com_utils
from sklearn.metrics import f1_score

# init params
comConfig = config.ComConfig()
fileConfig = config.FileConfig()
fasttextConfig = config.FastTextConfig()
if True:
    jieba.load_userdict(fileConfig.dir_jieba + fileConfig.file_jieba_dict)


def train():
    # train_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_train_data, mode='r', encoding='utf-8')
    # train_lines = []
    # for line in train_file:
    #     train_lines.append(line)
    print("start train fasttext model")
    model = fastText.train_unsupervised(input=fileConfig.dir_fasttext + fileConfig.file_fasttext_train_data,
                                        model=fasttextConfig.model_skipgram,
                                        dim=256, minCount=3, wordNgrams=6, thread=8, epoch=10)
    model.save_model(fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.model_skipgram))


def fasttext_get_sim(model, entity_text, mention_text, stopwords):
    true_text = data_utils.get_jieba_split_words(entity_text, jieba, stopwords)
    pre_text = data_utils.get_jieba_split_words(mention_text, jieba, stopwords)
    if len(true_text) == 0 or len(pre_text) == 0:
        return 0.0
    v1 = [model.get_word_vector(word=word) for word in true_text]
    v2 = [model.get_word_vector(word=word) for word in pre_text]
    score = np.dot(gensim.matutils.unitvec(np.array(v1).mean(axis=0)),
                   gensim.matutils.unitvec(np.array(v2).mean(axis=0)))
    return score


def acc_f1(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    f1 = f1_score(y_true, y_pred, average="macro")
    correct = np.sum((y_true == y_pred).astype(int))
    acc = correct / y_pred.shape[0]
    return acc, f1


def get_neighbor_sentence(sentence, mention_text):
    text_len = len(mention_text)
    neighbor_sentence = ''
    for i in range(len(sentence) - text_len + 1):
        if sentence[i:i + text_len] == mention_text:
            if i > 10 and i + text_len < len(sentence) - 9:
                neighbor_sentence = sentence[i - 10:i + text_len + 9]
            elif i < 10:
                neighbor_sentence = sentence[:20]
            elif i + text_len > len(sentence) - 9:
                neighbor_sentence = sentence[-20:]
    return neighbor_sentence


def test():
    print("start use the fasttext model to predict test data")
    if not os.path.exists(fileConfig.dir_result):
        os.mkdir(fileConfig.dir_result)
    model = fastText.load_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.model_skipgram))
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    dev_file = open(fileConfig.dir_ner + fileConfig.file_ner_test_cands_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_result + fileConfig.file_result_fasttext_test, 'w', encoding='utf-8')
    # entity diambiguation
    for line in tqdm(dev_file, 'entity diambiguation'):
        jstr = ujson.loads(line)
        dev_entity = {}
        text = jstr['text']
        dev_entity['text_id'] = jstr['text_id']
        dev_entity['text'] = jstr['text']
        mention_data = jstr['mention_data']
        mentions = []
        for mention in mention_data:
            mention_text = mention['mention']
            cands = mention['cands']
            if len(cands) == 0:
                continue
            # if len(cands) == 1:
            #     mentions.append(
            #         {'kb_id': str(cands[0]['cand_id']), 'mention': mention['mention'],
            #          'offset': str(mention['offset'])})
            #     continue
            max_index = 0
            max_score = 0.0
            # mention_neighbor_sentence = get_neighbor_sentence(text, mention_text)
            mention_neighbor_sentence = text
            for i, cand in enumerate(cands):
                score = fasttext_get_sim(model, mention_neighbor_sentence, cand['cand_text'], stopwords)
                if score > max_score:
                    max_score = score
                    max_index = i
            if max_score < fasttextConfig.min_entity_similarity_threshold:
                continue
            mentions.append(
                {'kb_id': cands[max_index]['cand_id'], 'mention': mention['mention'], 'offset': mention['offset']})
        dev_entity['mention_data'] = mentions
        dev_entity['mention_data_original'] = jstr['mention_data_original']
        out_file.write('-' * 20)
        out_file.write('\n')
        out_file.write("text_id:{}--text:{}".format(dev_entity['text_id'], dev_entity['text']))
        out_file.write('\n')
        out_file.write("mention_data:")
        out_file.write('\n')
        # generate mention
        for mention in dev_entity['mention_data']:
            kb_mention = ''
            if mention['kb_id'] != 'NIL':
                kb_mention = ujson.dumps(kb_dict[mention['kb_id']], ensure_ascii=False)
            out_file.write('*' * 20)
            out_file.write('\n')
            out_file.write('mention_original: {}'.format(mention))
            out_file.write('\n')
            out_file.write("kb: {}".format(kb_mention))
            out_file.write('\n')
            out_file.write('*' * 20)
            out_file.write('\n')
        # original mention
        out_file.write("kb_data:")
        out_file.write('\n')
        for mention in dev_entity['mention_data_original']:
            kb_mention = ''
            if mention['kb_id'] != 'NIL':
                kb_mention = ujson.dumps(kb_dict[mention['kb_id']], ensure_ascii=False)
            out_file.write('*' * 20)
            out_file.write('\n')
            out_file.write('kb_original: {}'.format(mention))
            out_file.write('\n')
            out_file.write("kb: {}".format(kb_mention))
            out_file.write('\n')
            out_file.write('*' * 20)
            out_file.write('\n')
        out_file.write('-' * 20)
        out_file.write('\n')
    print("success create test result")


def predict():
    print("start use the fasttext model to predict dev data")
    if not os.path.exists(fileConfig.dir_result):
        os.mkdir(fileConfig.dir_result)
    model = fastText.load_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.model_skipgram))
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    dev_file = open(fileConfig.dir_ner + fileConfig.file_ner_dev_cands_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_result + fileConfig.file_result_fasttext_predict, 'w', encoding='utf-8')
    # entity diambiguation
    for line in tqdm(dev_file, 'entity diambiguation'):
        jstr = ujson.loads(line)
        dev_entity = {}
        text = jstr['text']
        dev_entity['text_id'] = jstr['text_id']
        dev_entity['text'] = jstr['text']
        mention_data = jstr['mention_data']
        mentions = []
        for mention in mention_data:
            mention_text = mention['mention']
            cands = mention['cands']
            if len(cands) == 0:
                continue
            if len(cands) == 1:
                mentions.append(
                    {'kb_id': str(cands[0]['cand_id']), 'mention': mention['mention'],
                     'offset': str(mention['offset'])})
                continue
            max_index = 0
            max_score = 0.0
            # mention_neighbor_sentence = get_neighbor_sentence(text, mention_text)
            mention_neighbor_sentence = text
            for i, cand in enumerate(cands):
                score = fasttext_get_sim(model, mention_neighbor_sentence, cand['cand_text'], stopwords)
                if score > max_score:
                    max_score = score
                    max_index = i
            if max_score < fasttextConfig.min_entity_similarity_threshold:
                continue
            mentions.append(
                {'kb_id': cands[max_index]['cand_id'], 'mention': mention['mention'], 'offset': mention['offset']})
        # filter mentions
        choose_offset = {}
        for i, mention in enumerate(mentions):
            mention_offset = mention['offset']
            if choose_offset.get(mention_offset) is not None:
                if len(choose_offset.get(mention_offset).split('-')[1]) < len(mention['mention']):
                    choose_offset[mention_offset] = str(i) + '-' + mention['mention']
            else:
                choose_offset[mention_offset] = str(i) + '-' + mention['mention']
        choose_mentions = []
        for key, value in choose_offset.items():
            choose_mentions.append(mentions[int(value.split('-')[0])])
        dev_entity['mention_data'] = choose_mentions
        out_file.write(ujson.dumps(dev_entity, ensure_ascii=False))
        out_file.write('\n')
    print("success create predict result")


if __name__ == '__main__':
    if len(sys.argv) == 1 or not sys.argv[1] in ['train', 'test', 'predict']:
        print("should input param [train/test/predict]")
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'predict':
        predict()
    else:
        pass
