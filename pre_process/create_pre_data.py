import config
import pandas as pd
import multiprocessing
from utils import com_utils, data_utils
import ujson
import os
import random
import jieba_fast as jieba
from tqdm import tqdm
from collections import Counter, defaultdict

# init config
fileConfig = config.FileConfig()
nerConfig = config.NERConfig()
comConfig = config.ComConfig()
fasttextConfig = config.FastTextConfig()
if False:
    jieba.load_userdict(fileConfig.dir_jieba + fileConfig.file_jieba_dict)


def create_jieba_dict():
    data_file = open(fileConfig.dir_data + fileConfig.file_kb_data, mode='r', encoding='utf-8')
    com_utils.check_dir(fileConfig.dir_jieba)
    out_file = open(fileConfig.dir_jieba + fileConfig.file_jieba_dict, 'w', encoding='utf-8')
    words = set()
    for line in data_file:
        jstr = ujson.loads(line)
        subject = jstr['subject'].strip()
        words.add(subject)
        alias = jstr['alias']
        for item in alias:
            words.add(item.strip())
    # save file
    save_str = ''
    count = 0
    for word in tqdm(words):
        save_str += word + '\n'
        count += 1
        if count % 100 == 0:
            out_file.write(save_str)
            save_str = ''
    if len(save_str) > 0:
        print("write remid str")
        out_file.write(save_str)
    print("success build jieba dict")


def create_kb_dict():
    if not os.path.exists(fileConfig.dir_kb_info):
        os.mkdir(fileConfig.dir_kb_info)
    kb_datas = [line for line in
                open(fileConfig.dir_data + fileConfig.file_kb_data, mode='r', encoding='utf-8').readlines()]
    kb_dict = {}
    for kb_data in tqdm(kb_datas, desc='init kb dict'):
        kb_data = ujson.loads(kb_data)
        subject_id = kb_data['subject_id']
        if subject_id in kb_dict:
            raise Exception('key : {} exist'.format(subject_id))
        # text = data_utils.get_text(kb_data['data'], kb_data['subject'])
        all_alias = {}
        subject = com_utils.cht_to_chs(kb_data['subject'])
        alias = kb_data['alias']
        all_alias = com_utils.dict_add(all_alias, subject)
        for alia in alias:
            alia_text = com_utils.cht_to_chs(alia)
            if all_alias.get(alia_text) is not None:
                continue
            all_alias = com_utils.dict_add(all_alias, alia_text)
        text = data_utils.get_all_text(kb_data['subject'], kb_data['data'])
        kb_dict[subject_id] = {'type': kb_data['type'], 'subject': subject, 'alias': list(all_alias),
                               'text': text}
    com_utils.pickle_save(kb_dict, fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    print("create kb dict success")


def create_vocab():
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, mode='r', encoding='utf-8')
    dev_file = open(fileConfig.dir_data + fileConfig.file_dev_data, mode='r', encoding='utf-8')
    out_file = open(fileConfig.dir_data + fileConfig.file_vocab_data, mode='w', encoding='utf-8')
    vocab_dict = {}
    for line in tqdm(train_file, desc='deal train file'):
        jstr = ujson.loads(line)
        text = jstr['text']
        words = text.strip()
        for word in words:
            if word not in vocab_dict:
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1
    for line in tqdm(dev_file, desc='deal dev file'):
        jstr = ujson.loads(line)
        text = jstr['text']
        words = text.strip()
        for word in words:
            if word not in vocab_dict:
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1
    out_file.write('[PAD]\n')
    out_file.write('[UNK]\n')
    vocab_length = len(vocab_dict)
    for i, item in enumerate(Counter(vocab_dict).most_common()):
        out_file.write(item[0] + '\n') if i < vocab_length - 1 else out_file.write(item[0])
    out_file.close()
    print("success create vocab data")


def create_extend_train_file():
    print("start create extend train file...")
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, 'r', encoding='utf-8')
    extend_out_file = open(fileConfig.dir_data + fileConfig.file_extend_train_data, 'w', encoding='utf-8')
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    for line in tqdm(train_file, desc='extend train file'):
        jstr = ujson.loads(line)
        extend_lines = data_utils.get_extend_ner_train_list(kb_dict, jstr)
        print("test")


def create_ner_data(train_file_path=None):
    if not os.path.exists(fileConfig.dir_ner):
        os.mkdir(fileConfig.dir_ner)
    train_file = open(train_file_path, mode='r', encoding='utf-8')
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    data_list = []
    for i, line in tqdm(enumerate(train_file), desc='create ner data'):
        jstr = ujson.loads(line)
        text_id = jstr['text_id']
        text_list = list(jstr['text'])
        mentions = jstr['mention_data']
        text_len = len(text_list)
        tag_list = [nerConfig.O_seg] * text_len
        for mention in mentions:
            kb_id = mention['kb_id']
            if kb_id == 'NIL':
                continue
            mention_len = len(mention['mention'])
            offset = int(mention['offset'])
            # tag = nerConfig.NIL_seg if mention['kb_id'] == nerConfig.NIL_seg else nerConfig.KB_seg
            # tag = nerConfig.KB_seg
            # tag = com_utils.get_kb_type(kb_dict[kb_id]['type'])
            tag = nerConfig.KB_seg
            # tag B
            tag_list[offset] = nerConfig.B_seg + tag
            if mention_len == 1:
                continue
            # tag I
            for i in range(offset + 1, offset + mention_len - 1):
                tag_list[i] = nerConfig.I_seg + tag
            # tag E
            tag_list[offset + mention_len - 1] = nerConfig.E_seg + tag
        data_list.append({'id': text_id, 'text': text_list, 'tag': tag_list, 'mention_data': mentions})
    com_utils.pickle_save(data_list, fileConfig.dir_ner + fileConfig.file_ner_data)
    print("success create ner data")


def split_train_data():
    data_list = com_utils.pickle_load(fileConfig.dir_ner + fileConfig.file_ner_data)
    data_len = len(data_list)
    # train_size = int(data_len * comConfig.train_ratio)
    train_size = 80000
    random.seed(comConfig.random_seed)
    random.shuffle(data_list)

    # train_data = data_list[:train_size]
    train_data = data_list
    dev_data = data_list[train_size:]
    com_utils.pickle_save(train_data, fileConfig.dir_ner + fileConfig.file_ner_train_data)
    com_utils.pickle_save(dev_data, fileConfig.dir_ner + fileConfig.file_ner_dev_data)
    print("success split data set")


def create_predict_ner_data():
    if not os.path.exists(fileConfig.dir_ner):
        os.mkdir(fileConfig.dir_ner)
    train_file = open(fileConfig.dir_data + fileConfig.file_dev_data, mode='r', encoding='utf-8')
    data_list = []
    for i, line in tqdm(enumerate(train_file), desc='create ner data'):
        jstr = ujson.loads(line)
        text_id = jstr['text_id']
        text_list = list(jstr['text'])
        text_lenth = len(text_list)
        tag_list = [nerConfig.O_seg] * text_lenth
        data_list.append({'id': text_id, 'text': text_list, 'tag': tag_list})
    com_utils.pickle_save(data_list, fileConfig.dir_ner + fileConfig.file_ner_predict_data)
    print("success create ner dev data")


def create_nel_train_data():
    if not os.path.exists(fileConfig.dir_nel):
        os.mkdir(fileConfig.dir_nel)
    train_data = open(fileConfig.dir_data + fileConfig.file_train_data, 'r')
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    pd_df = pd.read_csv(fileConfig.dir_kb_info + fileConfig.file_kb_pandas_csv)
    data_list = []
    for line in tqdm(train_data, desc='create entity link train data'):
        # for line in train_data:
        jstr = ujson.loads(line)
        text_id = jstr['text_id']
        text = jstr['text']
        mention_datas = jstr['mention_data']
        for mention_data in mention_datas:
            kb_id = mention_data['kb_id']
            mention = mention_data['mention']
            start = mention_data['offset']
            end = int(start) + len(mention) - 1
            kb_entity = kb_dict.get(kb_id)
            if kb_entity is not None:
                entity_cands, entity_ids, entity_text = data_utils.get_entity_cands(kb_entity, kb_id, pd_df)
            else:
                continue
            data_list.append({'text_id': text_id, 'mention_text': text, 'mention': mention,
                              'mention_position': [start, end], 'entity_cands': entity_cands,
                              'entity_text': entity_text, 'entity_ids': entity_ids})
    com_utils.pickle_save(data_list, fileConfig.dir_nel + fileConfig.file_nel_entity_link_train_data)
    print("success create nel entity link train data")


def create_nel_vocab():
    # create mention entity vocab
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, mode='r', encoding='utf-8')
    dev_file = open(fileConfig.dir_data + fileConfig.file_dev_data, mode='r', encoding='utf-8')
    out_file = open(fileConfig.dir_nel + fileConfig.file_nel_mention_context_vocab, mode='w', encoding='utf-8')
    vocab_dict = {}
    for line in tqdm(train_file, desc='deal train file'):
        jstr = ujson.loads(line)
        text = jstr['text']
        words = text.strip()
        for word in words:
            if word not in vocab_dict:
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1
    for line in tqdm(dev_file, desc='deal dev file'):
        jstr = ujson.loads(line)
        text = jstr['text']
        words = text.strip()
        for word in words:
            if word not in vocab_dict:
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1
    out_file.write('[PAD]\n')
    out_file.write('[UNK]\n')
    vocab_length = len(vocab_dict)
    for i, item in enumerate(Counter(vocab_dict).most_common()):
        out_file.write(item[0] + '\n') if i < vocab_length - 1 else out_file.write(item[0])
    out_file.close()
    print("success create mention entity vocab data")

    # create entity context vocab
    entity_dict = set()
    text_dict = defaultdict(int)
    kb_data = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    for key, value in kb_data.items():
        text = value['text']
        subject = key
        if subject not in entity_dict:
            entity_dict.add(subject)
        else:
            raise Exception(f'entity : {subject} duplicated!!!')
        for t in text:
            text_dict[t] += 1
    entity_vocab = open(fileConfig.dir_nel + fileConfig.file_nel_entity_vocab, 'w', encoding='utf-8')
    entity_context_vocab = open(fileConfig.dir_nel + fileConfig.file_nel_entity_context_vocab, 'w', encoding='utf-8')
    for entity in entity_dict:
        entity_vocab.write(entity + '\n')
    entity_context_vocab.write('[PAD]\n')
    entity_context_vocab.write('[UNK]\n')
    for token, value in Counter(text_dict).most_common():
        entity_context_vocab.write(token + '\n')
    print("success create entity context vocab / entity vacab data")


def create_pandas_kb_data():
    kb_file = open(fileConfig.dir_data + fileConfig.file_kb_data, 'r', encoding='utf-8')
    subject_id_list = []
    subject_list = []
    type_list = []
    data_list = []
    for line in tqdm(kb_file, desc='deal kb file'):
        jstr = ujson.loads(line)
        subject_id_list.append(jstr['subject_id'])
        subject_list.append(com_utils.cht_to_chs(jstr['subject']))
        type_list.append(jstr['type'])
        data_list.append(jstr['data'])
    pandas_dict = {'subject_id': subject_id_list, 'subject': subject_list, 'type': type_list,
                   'data': data_list}
    df = pd.DataFrame.from_dict(pandas_dict)
    df.to_csv(fileConfig.dir_kb_info + fileConfig.file_kb_pandas_csv)
    print("success create pandas kb file")


def create_fasttext_unsup_train_data():
    if not os.path.exists(fileConfig.dir_fasttext):
        os.mkdir(fileConfig.dir_fasttext)
    kb_datas = open(fileConfig.dir_data + fileConfig.file_kb_data, 'r', encoding='utf-8')
    train_datas = open(fileConfig.dir_data + fileConfig.file_train_data, 'r', encoding='utf-8')
    dev_datas = open(fileConfig.dir_data + fileConfig.file_dev_data, 'r', encoding='utf-8')
    out_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_unsup_train_data, 'w', encoding='utf-8')
    stopword_list = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    print("prepare train data")
    train_sentence = []
    # kb data
    for line in tqdm(kb_datas, desc='deal kb data'):
        jstr = ujson.loads(line)
        texts = data_utils.get_kb_text_list(jstr, is_all=True)
        for text in texts:
            text_list = jieba.lcut(text)
            save_str = ''
            for text in text_list:
                if stopword_list.get(text) is None:
                    save_str += com_utils.cht_to_chs(text) + ' '
            train_sentence.append(save_str)
    # train data
    for line in tqdm(train_datas, desc='deal train data'):
        jstr = ujson.loads(line)
        text_list = jieba.lcut(jstr['text'])
        save_str = ''
        for text in text_list:
            if stopword_list.get(text) is None:
                save_str += com_utils.cht_to_chs(text) + ' '
        train_sentence.append(save_str)
    # dev data
    for line in tqdm(dev_datas, desc='deal dev data'):
        jstr = ujson.loads(line)
        text_list = jieba.lcut(jstr['text'])
        save_str = ''
        for text in text_list:
            if stopword_list.get(text) is None:
                save_str += com_utils.cht_to_chs(text) + ' '
        train_sentence.append(save_str)
    line_len = len(train_sentence)
    print("save train data")
    for i, line in enumerate(train_sentence):
        if i < line_len - 1:
            out_file.writelines(line)
            out_file.write('\n')
        else:
            out_file.writelines(line)
    print("success save fasttext train file")


def create_fasttext_sup_train_data(index, train_data_file, kb_dict_file, kb_alia_file, stopword_file, out_file):
    print("create {} sup train data".format(index))
    kb_alias_df = pd.read_csv(kb_alia_file)
    stopwords = data_utils.get_stopword_list(stopword_file)
    train_datas = open(train_data_file, 'r', encoding='utf-8')
    kb_dict = com_utils.pickle_load(kb_dict_file)
    train_out_file = open(out_file, 'w', encoding='utf-8')
    for line in tqdm(train_datas, desc='deal {} train file'.format(index)):
        jstr = ujson.loads(line)
        text = jstr['text']
        mentions = jstr['mention_data']
        for mention in mentions:
            mention_id = mention['kb_id']
            mention_text = mention['mention']
            neighbor_text = com_utils.get_neighbor_sentence(text, mention_text)
            # true values
            kb_entity = kb_dict.get(mention_id)
            if kb_entity is not None:
                out_str = com_utils.get_entity_mention_pair_text(kb_entity['text'], neighbor_text, stopwords,
                                                                 fasttextConfig.label_true)
                train_out_file.write(out_str)
            # false values
            alia_id = None
            alias_df = kb_alias_df[kb_alias_df['subject'] == com_utils.cht_to_chs(mention_text)]
            for _, item in alias_df.iterrows():
                a_id = str(item['subject_id'])
                if a_id != mention_id:
                    alia_id = a_id
                    break
            if alia_id is not None:
                alia_entity = kb_dict.get(alia_id)
                if alia_entity is not None:
                    out_str = com_utils.get_entity_mention_pair_text(kb_entity['text'], neighbor_text, stopwords,
                                                                     fasttextConfig.label_false)
                    train_out_file.write(out_str)


def test(index):
    print(index)


def create_fasttext_sup_data_multi():
    if not os.path.exists(fileConfig.dir_tmp):
        os.mkdir(fileConfig.dir_tmp)
    split_num = 8
    pool = multiprocessing.Pool(processes=split_num)
    train_data = open(fileConfig.dir_data + fileConfig.file_train_data, 'r', encoding='utf-8').readlines()
    if not os.path.exists(fileConfig.dir_tmp + fileConfig.file_train_data_split.format(split_num)):
        train_split_len = len(train_data) / split_num
        print("create train tmp file")
        for i in range(1, split_num + 1):
            train_data_split_file = open(fileConfig.dir_tmp + fileConfig.file_train_data_split.format(i), 'w',
                                         encoding='utf-8')
            train_datas = train_data[int((i - 1) * train_split_len):int(i * train_split_len)]
            train_data_split_file.writelines(train_datas)
    print('start create fasttext train data')
    if not os.path.exists(fileConfig.dir_tmp + fileConfig.file_fasttext_sup_train_split.format(split_num)):
        for index in range(1, split_num + 1):
            kb_alia_file = fileConfig.dir_kb_split + fileConfig.file_kb_pandas_alias_split.format(index)
            pool.apply_async(create_fasttext_sup_train_data,
                             (index, fileConfig.dir_tmp + fileConfig.file_train_data_split.format(index),
                              fileConfig.dir_kb_info + fileConfig.file_kb_dict,
                              kb_alia_file, fileConfig.dir_stopword + fileConfig.file_stopword,
                              fileConfig.dir_tmp + fileConfig.file_fasttext_sup_train_split.format(index),))
        pool.close()
        pool.join()  # behind close() or terminate()
        print("Sub-process(es) done.")
    if not os.path.exists(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_train_data):
        print("merge split train file")
        all_datas = []
        train_out_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_train_data, 'w', encoding='utf-8')
        for index in range(1, split_num + 1):
            split_file = open(fileConfig.dir_tmp + fileConfig.file_fasttext_sup_train_split.format(index), 'r',
                              encoding='utf-8')
            split_datas = split_file.readlines()
            all_datas += split_datas
            train_out_file.writelines(split_datas)
    if not os.path.exists(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_test_data):
        print("create test file...")
        test_out_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_test_data, 'w', encoding='utf-8')
        random.seed(comConfig.random_seed)
        random.shuffle(all_datas)
        test_out_lines = all_datas[0:20000]
        test_out_file.writelines(test_out_lines)
    print("success create fasttext sup train and test data")


def create_stopword():
    files = os.listdir(fileConfig.dir_stopword)
    out_file = open(fileConfig.dir_stopword + fileConfig.file_stopword, 'w', encoding='utf-8')
    stopword_set = set()
    for item in files:
        file = open(fileConfig.dir_stopword + item, 'r', encoding='utf-8')
        for line in file:
            stopword_set.add(line.strip().strip('\n').strip('\t'))
    length = len(stopword_set)
    for i, word in enumerate(stopword_set):
        if len(word) == 0:
            continue
        if i < length - 1:
            out_file.write(word)
            out_file.write('\n')
        else:
            out_file.write(word)
    print("success create stop word file")


def create_pandas_kb_alias_data():
    kb_file = open(fileConfig.dir_data + fileConfig.file_kb_data, 'r', encoding='utf-8')
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, 'r', encoding='utf-8')
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    subject_id_list = []
    subject_list = []
    subjects = {}
    # from kb file
    for line in tqdm(kb_file, desc='deal kb_file'):
        jstr = ujson.loads(line)
        subject_id = jstr['subject_id']
        subject = jstr['subject']
        subject_id_list.append(subject_id)
        subject_list.append(subject)
        alias = jstr['alias']
        subjects[subject] = 1
        for alia in alias:
            alia_str = com_utils.cht_to_chs(alia.strip().lower())
            if subjects.get(alia_str) is not None:
                continue
            else:
                subjects[alia_str] = 1
                subject_id_list.append(subject_id)
                subject_list.append(alia_str)
    # from train file
    for line in tqdm(train_file, desc='deal train file'):
        jstr = ujson.loads(line)
        mention_data = jstr['mention_data']
        for mention in mention_data:
            mention_text = mention['mention']
            mention_text = com_utils.cht_to_chs(mention_text)
            kb_id = mention['kb_id']
            kb_entity = kb_dict.get(kb_id)
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
                    if subjects.get(mention_text) is not None:
                        continue
                    else:
                        subjects[mention_text] = 1
                        subject_id_list.append(kb_id)
                        subject_list.append(mention_text)
    pandas_dict = {'subject_id': subject_id_list, 'subject': subject_list}
    df = pd.DataFrame.from_dict(pandas_dict)
    df.to_csv(fileConfig.dir_kb_info + fileConfig.file_kb_pandas_alias_data)
    print("success create pandas kb alia data file")


def create_tag_list():
    # ner_data = com_utils.pickle_load(fileConfig.dir_ner + fileConfig.file_ner_data)
    # tag_set = set()
    # for data in tqdm(ner_data, desc='create tag list'):
    #     for tag in data['tag']:
    #         tag_set.add(tag)
    # result_str = '['
    # for tag in tag_set:
    #     result_str += '\"' + tag + '\"' + ','
    # result_str = result_str[0:len(result_str) - 1]
    # result_str += ']'
    # print(result_str)

    tag_list = nerConfig.labels
    print(len(tag_list))
    for tag in tag_list:
        print(tag)


def fix_fasttext_sup_data():
    print('deal train file')
    train_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_train_data, 'r', encoding='utf-8')
    train_datas = train_file.readlines()
    train_chage_data = []
    for line in train_datas:
        if line.find('__true__') > -1:
            train_chage_data.append(line.replace('__true__', fasttextConfig.label_true))
        elif line.find('__false__') > -1:
            train_chage_data.append(line.replace('__false__', fasttextConfig.label_false))
    train_file.close()
    train_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_train_data, 'w', encoding='utf-8')
    train_file.writelines(train_chage_data)
    print('deal test file')
    test_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_test_data, 'r+', encoding='utf-8')
    test_datas = test_file.readlines()
    test_chage_data = []
    for line in test_datas:
        if line.find('__true__') > -1:
            test_chage_data.append(line.replace('__true__', fasttextConfig.label_true))
        elif line.find('__false__') > -1:
            test_chage_data.append(line.replace('__false__', fasttextConfig.label_false))
    test_file.close()
    test_file = open(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_test_data, 'w', encoding='utf-8')
    test_file.writelines(test_chage_data)
    print('success fixed')


if __name__ == '__main__':
    if not os.path.exists(fileConfig.dir_jieba + fileConfig.file_jieba_dict):
        create_jieba_dict()
    if not os.path.exists(fileConfig.dir_kb_info + fileConfig.file_kb_dict):
        create_kb_dict()
    if not os.path.exists(fileConfig.dir_data + fileConfig.file_vocab_data):
        create_vocab()
    create_extend_train_file()
    if not os.path.exists(fileConfig.dir_ner + fileConfig.file_ner_data):
        create_ner_data()
    if not os.path.exists(fileConfig.dir_ner + fileConfig.file_ner_train_data) or not os.path.exists(
            fileConfig.dir_ner + fileConfig.file_ner_dev_data):
        split_train_data()
    if not os.path.exists(fileConfig.dir_ner + fileConfig.file_ner_predict_data):
        create_predict_ner_data()
    if not os.path.exists(fileConfig.dir_nel + fileConfig.file_nel_entity_link_train_data):
        create_nel_train_data()
    if not os.path.exists(fileConfig.dir_nel + fileConfig.file_nel_entity_vocab) or not os.path.exists(
            fileConfig.dir_nel + fileConfig.file_nel_entity_context_vocab) or not os.path.exists(
        fileConfig.dir_nel + fileConfig.file_nel_mention_context_vocab):
        create_nel_vocab()
    if not os.path.exists(fileConfig.dir_kb_info + fileConfig.file_kb_pandas_csv):
        create_pandas_kb_data()
    if not os.path.exists(fileConfig.dir_kb_info + fileConfig.file_kb_pandas_alias_data):
        create_pandas_kb_alias_data()
    if not os.path.exists(fileConfig.dir_fasttext + fileConfig.file_fasttext_unsup_train_data):
        create_fasttext_unsup_train_data()
    if not os.path.exists(fileConfig.dir_fasttext + fileConfig.file_fasttext_sup_train_data):
        create_fasttext_sup_data_multi()
    if not os.path.exists(fileConfig.dir_stopword + fileConfig.file_stopword):
        create_stopword()
    # create_tag_list()
    # fix_fasttext_sup_data()
