import os
import re
import config
import pandas
import ast
import string
import ujson
import jieba_fast as jieba
import multiprocessing
from tqdm import tqdm
from utils import com_utils, data_utils

# init params
comConfig = config.ComConfig()
fileConfig = config.FileConfig()
nerConfig = config.NERConfig()

if True:
    jieba.load_userdict(fileConfig.dir_jieba + fileConfig.file_jieba_dict)


def get_jieba_mention(jieba_entities, jieba_char):
    if jieba_char in comConfig.punctuation:
        return None
    for entity in jieba_entities:
        if entity == jieba_char:
            return entity
    for entity in jieba_entities:
        if entity.find(jieba_char) > -1:
            return entity
    return None


def is_already_find_mention(mentions, text, offset):
    result = False
    for mention in mentions:
        if mention['mention'] == text and mention['offset'] == str(offset):
            result = True
            break
        if len(mention['mention']) == len(text) and len(mention['mention']) + int(mention['offset']) > offset:
            result = True
            break
    return result


def has_punctuation(text):
    result = False
    for c in text:
        if c in comConfig.punctuation:
            result = True
            break
    return result


def get_optim_mention_text(jieba_entities, mention_text):
    if has_punctuation(mention_text):
        for c in mention_text:
            if c not in comConfig.punctuation:
                for entity in jieba_entities:
                    if entity.find(c) > -1:
                        return entity
    else:
        return mention_text


def create_mention_data():
    mode = 1
    if mode == 1:
        create_dev_mention_data(mode, fileConfig.dir_ner + fileConfig.file_ner_predict_tag,
                                fileConfig.dir_ner + fileConfig.file_ner_dev_mention_data)
    elif mode == 2:
        create_dev_mention_data(mode, fileConfig.dir_ner + fileConfig.file_ner_test_predict_tag,
                                fileConfig.dir_ner + fileConfig.file_ner_test_mention_data)


# 获取offset
def get_offset(elem):
    return int(elem['offset'])


def create_dev_mention_data(mode, ner_datas, out_file):
    ner_datas = com_utils.pickle_load(ner_datas)
    text_id = 1
    dev_mention_data = []
    for data in tqdm(ner_datas, 'find entity'):
        text = ''.join(data['text'])
        tag_list = data['tag']
        start_index = 0
        mention_length = 0
        is_find = False
        mentions = []
        # use tag find
        for i, tag in enumerate(tag_list):
            if tag == nerConfig.B_seg + nerConfig.KB_seg:
                start_index = i
                mention_length += 1
                is_find = True
            elif tag == nerConfig.I_seg + nerConfig.KB_seg and is_find:
                mention_length += 1
            elif tag == nerConfig.E_seg + nerConfig.KB_seg and is_find:
                mention_length += 1
                mention = text[start_index:start_index + mention_length]
                mention = data_utils.strip_punctuation(mention)
                mentions.append({'mention': mention, 'offset': str(start_index)})
                is_find = False
                mention_length = 0
            elif tag == nerConfig.O_seg:
                is_find = False
        # use jieba find
        jieba_entities = jieba.lcut(text)
        for i, tag in enumerate(tag_list):
            if tag == nerConfig.B_seg + nerConfig.KB_seg or tag == nerConfig.I_seg + nerConfig.KB_seg or tag == nerConfig.E_seg + nerConfig.KB_seg:
                jieba_offset = i
                jieba_char = text[i]
                jieba_text = get_jieba_mention(jieba_entities, jieba_char)
                if jieba_text is None:
                    continue
                if not is_already_find_mention(mentions, jieba_text, jieba_offset):
                    mentions.append({'mention': jieba_text, 'offset': str(jieba_offset)})
        # sort mentions
        mentions.sort(key=get_offset)
        # optimize the mention data
        mentions_optim = []
        for mention in mentions:
            mentions_optim.append(
                {'mention': get_optim_mention_text(jieba_entities, mention['mention']), 'offset': mention['offset']})
        if mode == 1:
            dev_mention_data.append({'text_id': str(text_id), 'text': text, 'mention_data': mentions_optim})
        elif mode == 2:
            dev_mention_data.append({'text_id': str(text_id), 'text': text, 'mention_data': mentions_optim,
                                     'mention_data_original': data['mention_data_original']})
        text_id += 1
    com_utils.pickle_save(dev_mention_data, out_file)
    print("success create dev data with mentions, mode:{}".format(mode))


def split_dev_mention(num):
    dev_mention_data = com_utils.pickle_load(fileConfig.dir_ner + fileConfig.file_ner_dev_mention_data)
    data_len = len(dev_mention_data)
    block_size = data_len / num
    for i in range(1, num + 1):
        data_iter = dev_mention_data[int((i - 1) * block_size): int(i * block_size)]
        com_utils.pickle_save(data_iter, fileConfig.dir_ner_split + fileConfig.file_ner_dev_mention_split.format(i))
    print("success split dev mention to:{} files".format(num))


def split_test_mention(num):
    dev_mention_data = com_utils.pickle_load(fileConfig.dir_ner + fileConfig.file_ner_test_mention_data)
    data_len = len(dev_mention_data)
    block_size = data_len / num
    for i in range(1, num + 1):
        data_iter = dev_mention_data[int((i - 1) * block_size): int(i * block_size)]
        com_utils.pickle_save(data_iter, fileConfig.dir_ner_split + fileConfig.file_ner_test_mention_split.format(i))
    print("success split test mention to:{} files".format(num))


def is_find_from_alias(alias, mention_text):
    result = False
    for alia in alias:
        if alia.lower() == mention_text.lower():
            result = True
            break
    return result


def create_dev_mention_cands_data(index, mention_file, pd_file, alia_kb_df, out_file):
    print("start create {} mention cands".format(index))
    dev_mention_data = com_utils.pickle_load(mention_file)
    print("{} data length is {}".format(index, len(dev_mention_data)))
    pd_df = pandas.read_csv(pd_file)
    alia_kb_df = pandas.read_csv(alia_kb_df)
    for dev_data in tqdm(dev_mention_data, desc='find {} cands'.format(index)):
        mention_data = dev_data['mention_data']
        for mention in mention_data:
            mention_text = mention['mention']
            cands = []
            cand_ids = {}
            subject_df = pd_df[pd_df['subject'].str.lower() == mention_text.lower()]
            for _, item in subject_df.iterrows():
                s_id = str(item['subject_id'])
                if cand_ids.get(s_id) is not None:
                    continue
                cand_ids[s_id] = 1
                subject = item['subject']
                text = data_utils.get_text(ast.literal_eval(item['data']), item['subject'])
                cands.append({'cand_id': s_id, 'cand_subject': subject, 'cand_text': text})
            # match alias
            alias_subject_ids = []
            alias_df = alia_kb_df[alia_kb_df['subject'] == mention_text.lower()]
            for _, item in alias_df.iterrows():
                a_id = str(item['subject_id'])
                if alias_subject_ids.__contains__(a_id):
                    continue
                alias_subject_ids.append(a_id)
            for alia_id in alias_subject_ids:
                alias_df = pd_df[pd_df['subject_id'] == int(alia_id)]
                for _, item in alias_df.iterrows():
                    b_id = str(item['subject_id'])
                    if cand_ids.get(b_id) is not None:
                        continue
                    cand_ids[b_id] = 1
                    subject = item['subject']
                    text = data_utils.get_text(ast.literal_eval(item['data']), item['subject'])
                    cands.append({'cand_id': b_id, 'cand_subject': subject, 'cand_text': text})
            mention['cands'] = cands
    com_utils.pickle_save(dev_mention_data, out_file)
    print("success create {} dev data with mention and cands!".format(index))


def create_dev_mention_cands_multi(mention_file_name, out_file_name, num):
    pool = multiprocessing.Pool(processes=num)
    for i in range(1, num + 1):
        mention_file = fileConfig.dir_ner_split + mention_file_name.format(i)
        pd_file = fileConfig.dir_kb_split + fileConfig.file_kb_pandas_split.format(i)
        alia_pd_file = fileConfig.dir_kb_split + fileConfig.file_kb_pandas_alias_split.format(i)
        out_file = fileConfig.dir_ner_split + out_file_name.format(i)
        # print(mention_file)
        # print(pd_file)
        # print(out_file)
        pool.apply_async(create_dev_mention_cands_data, (i, mention_file, pd_file, alia_pd_file, out_file,))
    pool.close()
    pool.join()  # behind close() or terminate()
    print("Sub-process(es) done.")


def create_dev_cands_from_split(num):
    out_file = open(fileConfig.dir_ner + fileConfig.file_ner_dev_cands_data, 'w', encoding='utf-8')
    for i in range(1, num + 1):
        datas = com_utils.pickle_load(fileConfig.dir_ner_split + fileConfig.file_ner_dev_cands_split.format(i))
        for line in datas:
            text = ujson.dumps(line, ensure_ascii=False)
            out_file.write(text)
            out_file.write('\n')
    print("merge dev cands data success!")


def create_test_cands_from_split(num):
    out_file = open(fileConfig.dir_ner + fileConfig.file_ner_test_cands_data, 'w', encoding='utf-8')
    for i in range(1, num + 1):
        datas = com_utils.pickle_load(fileConfig.dir_ner_split + fileConfig.file_ner_test_cands_split.format(i))
        for line in datas:
            text = ujson.dumps(line, ensure_ascii=False)
            out_file.write(text)
            out_file.write('\n')
    print("merge test cands data success!")


if __name__ == '__main__':
    # if not os.path.exists(fileConfig.dir_ner + fileConfig.file_ner_dev_mention_data):
    #     create_dev_mention_data()
    # create_mention_data()
    split_num = 8
    # split_dev_mention(split_num)
    # split_test_mention(split_num)
    # create_dev_mention_cands_data()
    # create_dev_mention_cands_multi(mention_file_name=fileConfig.file_ner_dev_mention_split,
    #                                out_file_name=fileConfig.file_ner_dev_cands_split, num=split_num)
    # create_dev_mention_cands_multi(mention_file_name=fileConfig.file_ner_test_mention_split,
    #                                out_file_name=fileConfig.file_ner_test_cands_split, num=split_num)
    # index = 1
    # mention_file = fileConfig.dir_ner_split + fileConfig.file_ner_test_mention_split.format(index)
    # pd_file = fileConfig.dir_kb_split + fileConfig.file_kb_pandas_split.format(index)
    # alia_kb_file = fileConfig.dir_kb_info + fileConfig.file_kb_pandas_alias_data
    # out_file = fileConfig.dir_ner_split + fileConfig.file_ner_test_cands_split.format(index)
    # create_dev_mention_cands_data(index, mention_file, pd_file, alia_kb_file, out_file)
    # create_dev_cands_from_split(split_num)
    # create_test_cands_from_split(split_num)
