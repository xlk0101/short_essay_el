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
from tqdm import tqdm
from gensim.models import word2vec
from utils import data_utils, com_utils, text_cut
from sklearn.metrics import f1_score

# init params
comConfig = config.ComConfig()
fileConfig = config.FileConfig()
fasttextConfig = config.FastTextConfig()
cut_client = text_cut.get_client()


def train_unsup():
    # train_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_train_data, mode='r', encoding='utf-8')
    # train_lines = []
    # for line in train_file:
    #     train_lines.append(line)
    print("start train unsupervied fasttext model")
    model = fastText.train_unsupervised(input=fileConfig.dir_fasttext + fileConfig.file_fasttext_unsup_train_data,
                                        model=fasttextConfig.choose_model,
                                        dim=128, minCount=3, wordNgrams=7, minn=2, maxn=6, lr=0.1, thread=8, epoch=25,
                                        loss='hs')
    model.save_model(fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.choose_model))


def quantize():
    print("start quantize fasttext unsup model...")
    unsup_model = fastText.load_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.choose_model))
    unsup_model.quantize(input=fileConfig.dir_fasttext + fileConfig.file_fasttext_unsup_train_data, qnorm=True,
                         retrain=True, cutoff=300000)
    unsup_model.save_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_quantize_model.format(fasttextConfig.choose_model))


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def train_sup(mode=fasttextConfig.create_data_word):
    # train supervised model
    print('start train supervised fasttext model')
    # init path
    if mode == fasttextConfig.create_data_word:
        input_path = fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_train_word_data
        output_path = fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_word_model
        test_path = fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_test_word_data
    elif mode == fasttextConfig.create_data_char:
        input_path = fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_train_char_data
        output_path = fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_char_model
        test_path = fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_test_char_data
    # init path
    model = fastText.train_supervised(
        input=input_path, dim=200, epoch=100, lr=1.0,
        wordNgrams=3, ws=7, verbose=2, minCount=1, thread=8, loss='hs')
    print_results(*model.test(test_path))
    model.save_model(output_path)
    print("train sup fasttext finish")


def fasttext_get_sim(model, entity_text, mention_text, stopwords):
    true_text = data_utils.get_jieba_split_words(entity_text, cut_client, stopwords)
    pre_text = data_utils.get_jieba_split_words(mention_text, cut_client, stopwords)
    if len(true_text) == 0 or len(pre_text) == 0:
        return 0.0
    v1 = [model.get_word_vector(word=word) for word in true_text]
    v2 = [model.get_word_vector(word=word) for word in pre_text]
    score = np.dot(gensim.matutils.unitvec(np.array(v1).mean(axis=0)),
                   gensim.matutils.unitvec(np.array(v2).mean(axis=0)))
    return score


def get_gensim_wordvec(model, texts):
    result = []
    for text in texts:
        try:
            result.append(model.get_vector(text))
        except BaseException:
            pass
    return result


def gensim_get_sim(model, entity_text, mention_text, stopwords):
    true_text = data_utils.get_jieba_split_words(entity_text, cut_client, stopwords)
    pre_text = data_utils.get_jieba_split_words(mention_text, cut_client, stopwords)
    if len(true_text) == 0 or len(pre_text) == 0:
        return 0.0
    v1 = get_gensim_wordvec(model, true_text)
    v2 = get_gensim_wordvec(model, pre_text)
    if len(v1) == 0 or len(v2) == 0:
        return 0.0
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


def is_find_correct_entity(can_id, original_mention_data):
    result = False
    for original_mention in original_mention_data:
        original_id = original_mention['kb_id']
        if can_id == original_id:
            result = True
            break
    return result


def get_socre_key(item):
    return item['cand_score']


def test():
    print("start use the fasttext model to predict test data")
    if not os.path.exists(fileConfig.dir_result):
        os.mkdir(fileConfig.dir_result)
    model = fastText.load_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.choose_model))
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    dev_file = open(fileConfig.dir_ner + fileConfig.file_ner_test_cands_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_result + fileConfig.file_result_fasttext_test, 'w', encoding='utf-8')
    # f1 parmas
    gen_mention_count = 0
    original_mention_count = 0
    correct_mention_count = 0
    # entity diambiguation
    for line in tqdm(dev_file, 'entity diambiguation'):
        jstr = ujson.loads(line)
        dev_entity = {}
        text = jstr['text']
        dev_entity['text_id'] = jstr['text_id']
        dev_entity['text'] = jstr['text']
        mention_data = jstr['mention_data']
        original_mention_data = jstr['mention_data_original']
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
            max_cand = None
            # mention_neighbor_sentence = get_neighbor_sentence(text, mention_text)
            # score list
            score_list = []
            mention_neighbor_sentence = text
            for i, cand in enumerate(cands):
                score = fasttext_get_sim(model, mention_neighbor_sentence, cand['cand_text'], stopwords)
                # if score > max_score:
                #     max_score = score
                #     max_index = i
                if score < fasttextConfig.min_entity_similarity_threshold:
                    continue
                score_list.append({'cand_id': cand['cand_id'], 'cand_score': score, 'cand_type': cand['cand_type']})
            # if max_score < fasttextConfig.min_entity_similarity_threshold:
            #     continue
            # find the best cand
            find_type = False
            score_list.sort(key=get_socre_key, reverse=True)
            for item in score_list:
                if item['cand_type'] == mention['type']:
                    find_type = True
            if find_type:
                for item in score_list:
                    if item['cand_score'] > fasttextConfig.choose_entity_similarity_threshold:
                        max_cand = item
            if max_cand is None:
                if len(score_list) > 0:
                    max_cand = score_list[0]
            # find the best cand
            if max_cand is not None:
                if is_find_correct_entity(max_cand['cand_id'], original_mention_data):
                    correct_mention_count += 1
                mentions.append(
                    {'kb_id': max_cand['cand_id'], 'mention': mention['mention'], 'offset': mention['offset']})
        # calc f1 params
        gen_mention_count += len(mentions)
        original_mention_count += len(original_mention_data)

        dev_entity['mention_data'] = mentions
        dev_entity['mention_data_original'] = original_mention_data
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
    precision = correct_mention_count / gen_mention_count
    recall = correct_mention_count / original_mention_count
    f1 = 2 * precision * recall / (precision + recall)
    print("success create test result, p:{:.4f} r:{:.4f} f1:{:.4f}".format(precision, recall, f1))


def get_mention_len(item):
    return len(item['mention'])


def get_mention_offset(item):
    return int(item['offset'])


def test_sup(mode=fasttextConfig.create_data_word):
    print("start use the fasttext model/supervise model to predict test data")
    if not os.path.exists(fileConfig.dir_result):
        os.mkdir(fileConfig.dir_result)
    unsup_model_fasttext = fastText.load_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.choose_model))
    unsup_model_gensim = word2vec.Word2VecKeyedVectors.load(
        fileConfig.dir_fasttext + fileConfig.file_gensim_tencent_unsup_model)
    sup_model = fastText.load_model(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_word_model)
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    dev_file = open(fileConfig.dir_ner + fileConfig.file_ner_test_cands_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_result + fileConfig.file_result_fasttext_test, 'w', encoding='utf-8')
    # f1 parmas
    gen_mention_count = 0
    original_mention_count = 0
    correct_mention_count = 0
    # count = 0
    # entity diambiguation
    for line in tqdm(dev_file, 'entity diambiguation'):
        # count += 1
        # if count < 3456:
        #     continue
        jstr = ujson.loads(line)
        dev_entity = {}
        text = com_utils.cht_to_chs(jstr['text'].lower())
        dev_entity['text_id'] = jstr['text_id']
        dev_entity['text'] = jstr['text']
        mention_data = jstr['mention_data']
        original_mention_data = jstr['mention_data_original']
        mentions = []
        for mention in mention_data:
            mention_text = mention['mention']
            if mention_text is None:
                continue
            cands = mention['cands']
            if len(cands) == 0:
                continue
            # use supervised model to choose mention
            supervise_cands = []
            for cand in cands:
                neighbor_text = com_utils.get_neighbor_sentence(text, com_utils.cht_to_chs(mention_text.lower()))
                cand_entity = kb_dict.get(cand['cand_id'])
                if cand_entity is not None:
                    out_str = com_utils.get_entity_mention_pair_text(com_utils.cht_to_chs(cand_entity['text'].lower()),
                                                                     neighbor_text, stopwords, cut_client, mode=mode)
                    # print(out_str)
                    result = sup_model.predict(out_str.replace('\n', ' '))[0][0]
                    if result == fasttextConfig.label_true:
                        supervise_cands.append(cand)
            # unsupervise model choose item
            max_cand = None
            if len(supervise_cands) == 0:
                supervise_cands = cands
            # score list
            score_list = []
            mention_neighbor_sentence = text
            for i, cand in enumerate(supervise_cands):
                # score_fasttext = fasttext_get_sim(unsup_model_fasttext, mention_neighbor_sentence,
                #                          com_utils.cht_to_chs(cand['cand_text'].lower()), stopwords)
                score_gensim = gensim_get_sim(unsup_model_gensim, mention_neighbor_sentence,
                                              com_utils.cht_to_chs(cand['cand_text'].lower()), stopwords)
                # score = (0.8 * score_gensim) + (0.2 * score_fasttext)
                score = score_gensim
                # if score > max_score:
                #     max_score = score
                #     max_index = score
                if score < fasttextConfig.min_entity_similarity_threshold:
                    continue
                score_list.append({'cand_id': cand['cand_id'], 'cand_score': score, 'cand_type': cand['cand_type']})
            # if max_score < fasttextConfig.min_entity_similarity_threshold:
            #     continue
            # find the best cand
            # find_type = False
            score_list.sort(key=get_socre_key, reverse=True)
            # for item in score_list:
            #     if item['cand_type'] == mention['type']:
            #         find_type = True
            # if find_type:
            #     for item in score_list:
            #         if item['cand_score'] > fasttextConfig.choose_entity_similarity_threshold:
            #             max_cand = item
            if max_cand is None:
                if len(score_list) > 0:
                    max_cand = score_list[0]
            # find the best cand
            if max_cand is not None:
                mentions.append(
                    {'kb_id': max_cand['cand_id'], 'mention': mention['mention'], 'offset': mention['offset']})
        # optim mentions
        delete_mentions = []
        mentions.sort(key=get_mention_len)
        for mention in mentions:
            mention_offset = int(mention['offset'])
            mention_len = len(mention['mention'])
            for sub_mention in mentions:
                if mention_offset != int(sub_mention['offset']) and int(sub_mention['offset']) in range(mention_offset,
                                                                                                        mention_offset + mention_len):
                    if not data_utils.is_mention_already_in_list(delete_mentions, sub_mention):
                        delete_mentions.append(sub_mention)
                if mention_offset == int(sub_mention['offset']) and len(mention['mention']) > len(
                        sub_mention['mention']):
                    if not data_utils.is_mention_already_in_list(delete_mentions, sub_mention):
                        delete_mentions.append(sub_mention)
        if len(delete_mentions) > 0:
            change_mentions = []
            for mention in mentions:
                if not data_utils.is_mention_already_in_list(delete_mentions, mention):
                    change_mentions.append(mention)
            mentions = change_mentions
        change_mentions = []
        for mention in mentions:
            if not data_utils.is_mention_already_in_list(change_mentions, mention) and mention[
                'mention'] not in comConfig.punctuation:
                change_mentions.append(mention)
        mentions = change_mentions
        mentions.sort(key=get_mention_offset)
        # optim mentions
        # calc f1
        for mention in mentions:
            if is_find_correct_entity(mention['kb_id'], original_mention_data):
                correct_mention_count += 1
        gen_mention_count += len(mentions)
        for orginal_mention in original_mention_data:
            if orginal_mention['kb_id'] != 'NIL':
                original_mention_count += 1
        # out result
        dev_entity['mention_data'] = mentions
        dev_entity['mention_data_original'] = original_mention_data
        out_file.write(ujson.dumps(dev_entity, ensure_ascii=False))
        out_file.write('\n')
    precision = correct_mention_count / gen_mention_count
    recall = correct_mention_count / original_mention_count
    f1 = 2 * precision * recall / (precision + recall)
    print("success create test result, p:{:.4f} r:{:.4f} f1:{:.4f}".format(precision, recall, f1))


def predict():
    print("start use the fasttext model to predict dev data")
    if not os.path.exists(fileConfig.dir_result):
        os.mkdir(fileConfig.dir_result)
    model = fastText.load_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.choose_model))
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


def predict_sup(mode=fasttextConfig.create_data_word):
    print("start use the fasttext/supervised model to predict dev data")
    if not os.path.exists(fileConfig.dir_result):
        os.mkdir(fileConfig.dir_result)
    # unsup_model = fastText.load_model(
    #     fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.model_skipgram))
    unsup_model = word2vec.Word2VecKeyedVectors.load(
        fileConfig.dir_fasttext + fileConfig.file_gensim_tencent_unsup_model)
    sup_model = fastText.load_model(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_word_model)
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    dev_file = open(fileConfig.dir_ner + fileConfig.file_ner_dev_cands_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_result + fileConfig.file_result_fasttext_predict, 'w', encoding='utf-8')
    # entity diambiguation
    for line in tqdm(dev_file, 'entity diambiguation'):
        jstr = ujson.loads(line)
        dev_entity = {}
        text = com_utils.cht_to_chs(jstr['text'].lower())
        dev_entity['text_id'] = jstr['text_id']
        dev_entity['text'] = jstr['text']
        mention_data = jstr['mention_data']
        mentions = []
        for mention in mention_data:
            mention_text = mention['mention']
            if mention_text is None:
                continue
            cands = mention['cands']
            if len(cands) == 0:
                continue
            # use supervised model to choose mention
            supervise_cands = []
            for cand in cands:
                neighbor_text = com_utils.get_neighbor_sentence(text, com_utils.cht_to_chs(mention_text.lower()))
                cand_entity = kb_dict.get(cand['cand_id'])
                if cand_entity is not None:
                    out_str = com_utils.get_entity_mention_pair_text(com_utils.cht_to_chs(cand_entity['text'].lower()),
                                                                     neighbor_text, stopwords, cut_client, mode=mode)
                    result = sup_model.predict(out_str.strip('\n'))[0][0]
                    if result == fasttextConfig.label_true:
                        supervise_cands.append(cand)
            if len(supervise_cands) == 0:
                supervise_cands = cands
            # unsupervise model choose item
            max_cand = None
            # score list
            score_list = []
            mention_neighbor_sentence = text
            for i, cand in enumerate(supervise_cands):
                # score = fasttext_get_sim(unsup_model, mention_neighbor_sentence,
                #                          com_utils.cht_to_chs(cand['cand_text'].lower()), stopwords)
                score = gensim_get_sim(unsup_model, mention_neighbor_sentence,
                                       com_utils.cht_to_chs(cand['cand_text'].lower()), stopwords)
                if score < fasttextConfig.min_entity_similarity_threshold:
                    continue
                score_list.append({'cand_id': cand['cand_id'], 'cand_score': score, 'cand_type': cand['cand_type']})
            score_list.sort(key=get_socre_key, reverse=True)
            if len(score_list) > 0:
                max_cand = score_list[0]
            # find the best cand
            if max_cand is not None:
                mentions.append(
                    {'kb_id': max_cand['cand_id'], 'mention': mention['mention'], 'offset': mention['offset']})
        # optim mentions
        delete_mentions = []
        mentions.sort(key=get_mention_len)
        for optim_mention in mentions:
            mention_offset = int(optim_mention['offset'])
            mention_len = len(optim_mention['mention'])
            for sub_mention in mentions:
                if mention_offset != int(sub_mention['offset']) and int(sub_mention['offset']) in range(
                        mention_offset,
                        mention_offset + mention_len):
                    if not data_utils.is_mention_already_in_list(delete_mentions, sub_mention):
                        delete_mentions.append(sub_mention)
        if len(delete_mentions) > 0:
            change_mentions = []
            for optim_mention in mentions:
                if not data_utils.is_mention_already_in_list(delete_mentions, optim_mention):
                    change_mentions.append(optim_mention)
            mentions = change_mentions
        change_mentions = []
        for optim_mention in mentions:
            if not data_utils.is_mention_already_in_list(change_mentions, optim_mention) and optim_mention[
                'mention'] not in comConfig.punctuation:
                change_mentions.append(optim_mention)
        mentions = change_mentions
        mentions.sort(key=get_mention_offset)
        # optim mentions
        # filter mentions
        # choose_offset = {}
        # for i, mention in enumerate(mentions):
        #     mention_offset = mention['offset']
        #     if choose_offset.get(mention_offset) is not None:
        #         if len(choose_offset.get(mention_offset).split('-')[1]) < len(mention['mention']):
        #             choose_offset[mention_offset] = str(i) + '-' + mention['mention']
        #     else:
        #         choose_offset[mention_offset] = str(i) + '-' + mention['mention']
        # choose_mentions = []
        # for key, value in choose_offset.items():
        #     choose_mentions.append(mentions[int(value.split('-')[0])])
        # filter mentions
        dev_entity['mention_data'] = mentions
        out_file.write(ujson.dumps(dev_entity, ensure_ascii=False))
        out_file.write('\n')
    print("success create supervised predict result")


def eval_sup(mode=fasttextConfig.create_data_word):
    print("start use the fasttext/supervised model to predict eval data")
    if not os.path.exists(fileConfig.dir_result):
        os.mkdir(fileConfig.dir_result)
    # unsup_model = fastText.load_model(
    #     fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.model_skipgram))
    unsup_model = word2vec.Word2VecKeyedVectors.load(
        fileConfig.dir_fasttext + fileConfig.file_gensim_tencent_unsup_model)
    sup_model = fastText.load_model(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_word_model)
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    dev_file = open(fileConfig.dir_ner + fileConfig.file_ner_eval_cands_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_result + fileConfig.file_result_eval_data, 'w', encoding='utf-8')
    # entity diambiguation
    for line in tqdm(dev_file, 'entity diambiguation'):
        if len(line.strip('\n')) == 0:
            continue
        jstr = ujson.loads(line)
        dev_entity = {}
        text = com_utils.cht_to_chs(jstr['text'].lower())
        dev_entity['text_id'] = jstr['text_id']
        dev_entity['text'] = jstr['text']
        mention_data = jstr['mention_data']
        mentions = []
        for mention in mention_data:
            mention_text = mention['mention']
            if mention_text is None:
                continue
            cands = mention['cands']
            if len(cands) == 0:
                continue
            # use supervised model to choose mention
            supervise_cands = []
            for cand in cands:
                neighbor_text = com_utils.get_neighbor_sentence(text, com_utils.cht_to_chs(mention_text.lower()))
                cand_entity = kb_dict.get(cand['cand_id'])
                if cand_entity is not None:
                    out_str = com_utils.get_entity_mention_pair_text(com_utils.cht_to_chs(cand_entity['text'].lower()),
                                                                     neighbor_text, stopwords, cut_client, mode=mode)
                    result = sup_model.predict(out_str.strip('\n'))[0][0]
                    if result == fasttextConfig.label_true:
                        supervise_cands.append(cand)
            if len(supervise_cands) == 0:
                supervise_cands = cands
            # unsupervise model choose item
            max_cand = None
            # score list
            score_list = []
            mention_neighbor_sentence = text
            for i, cand in enumerate(supervise_cands):
                # score = fasttext_get_sim(unsup_model, mention_neighbor_sentence,
                #                          com_utils.cht_to_chs(cand['cand_text'].lower()), stopwords)
                score = gensim_get_sim(unsup_model, mention_neighbor_sentence,
                                       com_utils.cht_to_chs(cand['cand_text'].lower()), stopwords)
                if score < fasttextConfig.min_entity_similarity_threshold:
                    continue
                score_list.append({'cand_id': cand['cand_id'], 'cand_score': score, 'cand_type': cand['cand_type']})
            score_list.sort(key=get_socre_key, reverse=True)
            if len(score_list) > 0:
                max_cand = score_list[0]
            # find the best cand
            if max_cand is not None:
                mentions.append(
                    {'kb_id': max_cand['cand_id'], 'mention': mention['mention'], 'offset': mention['offset']})
        # optim mentions
        delete_mentions = []
        mentions.sort(key=get_mention_len)
        for optim_mention in mentions:
            mention_offset = int(optim_mention['offset'])
            mention_len = len(optim_mention['mention'])
            for sub_mention in mentions:
                if mention_offset != int(sub_mention['offset']) and int(sub_mention['offset']) in range(
                        mention_offset,
                        mention_offset + mention_len):
                    if not data_utils.is_mention_already_in_list(delete_mentions, sub_mention):
                        delete_mentions.append(sub_mention)
        if len(delete_mentions) > 0:
            change_mentions = []
            for optim_mention in mentions:
                if not data_utils.is_mention_already_in_list(delete_mentions, optim_mention):
                    change_mentions.append(optim_mention)
            mentions = change_mentions
        change_mentions = []
        for optim_mention in mentions:
            if not data_utils.is_mention_already_in_list(change_mentions, optim_mention) and optim_mention[
                'mention'] not in comConfig.punctuation:
                change_mentions.append(optim_mention)
        mentions = change_mentions
        mentions.sort(key=get_mention_offset)
        dev_entity['mention_data'] = mentions
        out_file.write(ujson.dumps(dev_entity, ensure_ascii=False))
        out_file.write('\n')
    print("success create supervised eval result")


if __name__ == '__main__':
    if len(sys.argv) == 1 or not sys.argv[1] in ['train_sup', 'train_unsup', 'test', 'test_sup', 'predict',
                                                 'predict_sup', 'eval_sup', 'quantize']:
        print("should input param [train/test/predict]")
    if sys.argv[1] == 'train_unsup':
        train_unsup()
    elif sys.argv[1] == 'train_sup':
        train_sup(mode=fasttextConfig.create_data_word)
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'test_sup':
        test_sup(mode=fasttextConfig.create_data_word)
    elif sys.argv[1] == 'predict':
        predict()
    elif sys.argv[1] == 'predict_sup':
        predict_sup(mode=fasttextConfig.create_data_word)
    elif sys.argv[1] == 'eval_sup':
        eval_sup()
    elif sys.argv[1] == 'quantize':
        quantize()
    else:
        pass
