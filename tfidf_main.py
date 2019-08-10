"""
use tfidf method to implement entity disambiguation
经验证该方法无用。。。
"""
import sys
import os
import config
import ujson
import random
import numpy as np
from tqdm import tqdm
from utils import com_utils, text_cut
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# init params
fileConfig = config.FileConfig()
tfidfConfig = config.TfIdfConfig()
comConfig = config.ComConfig()
cut_client = text_cut.get_client()


def train():
    datas = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    vectorizer = TfidfVectorizer()
    train_sentence = []
    print("prepare train data")
    for key, data in tqdm(datas.items(), desc='init train data'):
        train_sentence.append(' '.join(cut_client.cut_text(data['text'])))
    print("start train tfidf model")
    X = vectorizer.fit_transform(train_sentence)
    print("save model and keyword")
    tfidf_save_data = [X, vectorizer]
    if not os.path.exists(fileConfig.dir_tfidf):
        os.mkdir(fileConfig.dir_tfidf)
    com_utils.pickle_save(tfidf_save_data, fileConfig.dir_tfidf + fileConfig.file_tfidf_save_data)
    print("success train and save tfidf file")


# 获取包含关键词的句子中关键词所属的entity_id
def get_entityid(sentence, vectorizer, X):
    id_start = tfidfConfig.start_index
    a_list = [' '.join(cut_client.cut_text()(sentence))]
    res = cosine_similarity(vectorizer.transform(a_list), X)[0]
    top_idx = np.argsort(res)[-1]
    return id_start + top_idx


def acc_f1(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    f1 = f1_score(y_true, y_pred, average="macro")
    correct = np.sum((y_true == y_pred).astype(int))
    acc = correct / y_pred.shape[0]
    return acc, f1


def test():
    print("start test the tfidf model")
    tfidf_data = com_utils.pickle_load(fileConfig.dir_tfidf + fileConfig.file_tfidf_save_data)
    vectorizer = tfidf_data[1]
    X = tfidf_data[0]
    # init test data
    ratio = 0.01
    print("init test datas use ratio:{}".format(ratio))
    test_datas = []
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, 'r', encoding='utf-8')
    for line in train_file:
        test_datas.append(ujson.loads(line))
    test_data_len = int(len(test_datas) * ratio)
    random.seed(comConfig.random_seed)
    random.shuffle(test_datas)
    test_datas = test_datas[:test_data_len]
    mentions = []
    for data in test_datas:
        mention_datas = data['mention_data']
        for mention in mention_datas:
            if mention['kb_id'] != 'NIL':
                mention_copy = mention.copy()
                mention_copy['sentence'] = data['text']
                mentions.append(mention_copy)
    # start test model
    print("start find mention")
    y_pred = []
    y_true = []
    for mention in tqdm(mentions, desc='find mention'):
        # for mention in mentions:
        y_true.append(int(mention['kb_id']))
        text = mention['mention']
        text_len = len(text)
        sentence = mention['sentence']
        for i in range(len(sentence) - text_len + 1):
            if sentence[i:i + text_len] == text:
                if i > 10 and i + text_len < len(sentence) - 9:
                    neighbor_sentence = sentence[i - 10:i + text_len + 9]
                elif i < 10:
                    neighbor_sentence = sentence[:20]
                elif i + text_len > len(sentence) - 9:
                    neighbor_sentence = sentence[-20:]
                kb_id = get_entityid(neighbor_sentence, vectorizer, X)
                y_pred.append(kb_id)
                break
    # calc the f1
    acc, f1 = acc_f1(y_pred, y_true)
    print("acc:{:.4f} f1:{:.4f}".format(acc, f1))


if __name__ == '__main__':
    if len(sys.argv) == 1 or not sys.argv[1] in ['train', 'test']:
        print("should input param [train/test/predict]")
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        pass
