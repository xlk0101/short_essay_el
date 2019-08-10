import ast
import multiprocessing
import sys
from collections import Counter

import pandas
import ujson
from tqdm import tqdm
from gensim.models import word2vec

import config
from utils import com_utils, data_utils, text_cut

# init params
comConfig = config.ComConfig()
fileConfig = config.FileConfig()
nerConfig = config.NERConfig()
cut_client = text_cut.get_client()


def get_jieba_mention(jieba_entities, jieba_char, jieba_offset):
    if jieba_char in comConfig.punctuation:
        return None
    text_count = -1
    for entity in jieba_entities:
        for char in entity:
            text_count += 1
            if jieba_offset == text_count and jieba_char == char:
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
    # for c in text:
    #     if c in comConfig.punctuation:
    #         result = True
    #         break
    if text[0] in comConfig.punctuation:
        result = True
    elif text[len(text) - 1] in comConfig.punctuation:
        result = True
    return result


def get_optim_mention_text(jieba_entities, mention_text):
    # if has_punctuation(mention_text):
    #     for c in mention_text:
    #         if c not in comConfig.punctuation:
    #             for entity in jieba_entities:
    #                 if entity.find(c) > -1:
    #                     return entity
    # else:
    #     return mention_text
    return mention_text


def create_mention_data(mode):
    if mode == 1:
        create_dev_mention_data(mode, fileConfig.dir_ner + fileConfig.file_ner_predict_tag,
                                fileConfig.dir_ner + fileConfig.file_ner_dev_mention_data)
    elif mode == 2:
        create_dev_mention_data(mode, fileConfig.dir_ner + fileConfig.file_ner_test_predict_tag,
                                fileConfig.dir_ner + fileConfig.file_ner_test_mention_data)
    elif mode == 3:
        create_dev_mention_data(mode, fileConfig.dir_ner + fileConfig.file_ner_eval_predict_tag,
                                fileConfig.dir_ner + fileConfig.file_ner_eval_mention_data)


def get_simi_subject_list(simi_list):
    result = []
    for item in simi_list:
        result.append(item[0])
    return result


def gen_simi_subject_list(file_path):
    print('start gen similar subject...')
    file_datas = com_utils.pickle_load(file_path)
    gensim_model = word2vec.Word2VecKeyedVectors.load(
        fileConfig.dir_fasttext + fileConfig.file_gensim_tencent_unsup_model)
    for item in tqdm(file_datas, 'gen simi subject'):
        mention_data = item['mention_data']
        for mention in mention_data:
            mention_text = mention['mention']
            try:
                mention['gen_subjects'] = get_simi_subject_list(
                    gensim_model.most_similar(positive=[mention_text], topn=5))
            except BaseException:
                mention['gen_subjects'] = []
    com_utils.pickle_save(file_datas, file_path)
    print('success gen similar subject...')


# 获取offset
def get_offset(elem):
    return int(elem['offset'])


def get_mention_len(item):
    return len(item['mention'])


def create_dev_mention_data(mode, ner_datas, out_file):
    ner_datas = com_utils.pickle_load(ner_datas)
    jieba_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_jieba_kb)
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    gen_more_words = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_analysis_gen_more)
    text_id = 1
    dev_mention_data = []
    # count = 0
    for data in tqdm(ner_datas, 'find entity'):
        # count += 1
        # if count < 1496:
        #     continue
        text = ''.join(data['text'])
        tag_list = data['tag']
        start_index = 0
        mention_length = 0
        is_find = False
        mentions = []
        type_dict = {}
        # use tag find
        for i, tag in enumerate(tag_list):
            # if tag == nerConfig.B_seg + nerConfig.KB_seg:
            if tag.find(nerConfig.B_seg) > -1 or (tag.find(nerConfig.I_seg) > -1 and not is_find):
                type_str = tag.split('_')[1]
                type_dict = com_utils.dict_add(type_dict, type_str)
                start_index = i
                mention_length = 1
                is_find = True
            elif tag.find(nerConfig.E_seg) > -1 and not is_find:
                type_str = tag.split('_')[1]
                type_dict = com_utils.dict_add(type_dict, type_str)
                start_index = i
                mention_length += 1
                mention = text[start_index:start_index + mention_length]
                mention = data_utils.strip_punctuation(mention)
                type_list = Counter(type_dict).most_common()
                mentions.append({'mention': mention, 'offset': str(start_index), 'type': type_list[0][0]})
                is_find = False
                mention_length = 0
                type_dict = {}
            # elif tag == nerConfig.I_seg + nerConfig.KB_seg and is_find:
            elif tag.find(nerConfig.I_seg) > -1 and is_find:
                type_str = tag.split('_')[1]
                type_dict = com_utils.dict_add(type_dict, type_str)
                mention_length += 1
            # elif tag == nerConfig.E_seg + nerConfig.KB_seg and is_find:
            elif tag.find(nerConfig.E_seg) > -1 and is_find:
                type_str = tag.split('_')[1]
                type_dict = com_utils.dict_add(type_dict, type_str)
                mention_length += 1
                mention = text[start_index:start_index + mention_length]
                mention = data_utils.strip_punctuation(mention)
                type_list = Counter(type_dict).most_common()
                mentions.append({'mention': mention, 'offset': str(start_index), 'type': type_list[0][0]})
                is_find = False
                mention_length = 0
                type_dict = {}
            elif tag == nerConfig.O_seg:
                is_find = False
                mention_length = 0
                type_dict = {}
        # use jieba find
        jieba_entities = cut_client.cut_text(text)
        for i, tag in enumerate(tag_list):
            # if tag == nerConfig.B_seg + nerConfig.KB_seg or tag == nerConfig.I_seg + nerConfig.KB_seg or tag == nerConfig.E_seg + nerConfig.KB_seg:
            if tag.find(nerConfig.B_seg) > -1 or tag.find(nerConfig.I_seg) > -1 or tag.find(nerConfig.E_seg) > -1:
                jieba_offset = i
                jieba_char = text[i]
                jieba_text = get_jieba_mention(jieba_entities, jieba_char, jieba_offset)
                if jieba_text is None:
                    continue
                elif jieba_text == '_' or jieba_text == '-':
                    continue
                elif data_utils.is_punctuation(jieba_text):
                    continue
                elif len(jieba_text) == 1:
                    continue
                elif stopwords.get(jieba_text) is not None:
                    continue
                # elif gen_more_words.get(jieba_text) is not None:
                #     continue
                jieba_offset = jieba_offset - jieba_text.find(jieba_char)
                if len(jieba_text) <= comConfig.max_jieba_cut_len and (
                        jieba_dict.get(jieba_text) is not None):
                    type_str = tag.split('_')[1] if tag.find('_') > -1 else 'O'
                    if jieba_text is None:
                        continue
                    if not is_already_find_mention(mentions, jieba_text, jieba_offset):
                        mentions.append({'mention': jieba_text, 'offset': str(jieba_offset), 'type': type_str})
        # find inner brackets mentions
        bracket_mentions = data_utils.get_mention_inner_brackets(text, tag_list)
        if len(bracket_mentions) > 0:
            mentions += bracket_mentions
        # completion mentions
        # mentions_com = []
        # for mention in mentions:
        #     mention_str = mention['mention']
        #     try:
        #         for find in re.finditer(mention_str, text):
        #             find_offset = find.span()[0]
        #             if find_offset != int(mention['offset']):
        #                 mentions_com.append(
        #                     {'mention': mention['mention'], 'offset': str(find_offset), 'type': mention['type']})
        #     except BaseException:
        #         # print("occur error when match mention str in completion mentions, error value:{} text:{}".format(
        #         #     mention_str, text))
        #         pass
        #     mentions_com.append(mention)
        # mentions = mentions_com
        # optim mentions
        delete_mentions = []
        mentions.sort(key=get_mention_len)
        for mention in mentions:
            mention_offset = int(mention['offset'])
            mention_len = len(mention['mention'])
            for sub_mention in mentions:
                if mention_offset != int(sub_mention['offset']) and int(sub_mention['offset']) in range(
                        mention_offset,
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
        # optim mentions
        # sort mentions
        mentions.sort(key=get_offset)
        # optimize the mention data
        mentions_optim = []
        for mention in mentions:
            mentions_optim.append(
                {'mention': get_optim_mention_text(jieba_entities, mention['mention']), 'offset': mention['offset'],
                 'type': mention['type']})
        if mode == 1:
            dev_mention_data.append({'text_id': str(text_id), 'text': text, 'mention_data': mentions_optim})
        elif mode == 2:
            dev_mention_data.append({'text_id': str(text_id), 'text': text, 'mention_data': mentions_optim,
                                     'mention_data_original': data['mention_data_original']})
        elif mode == 3:
            dev_mention_data.append({'text_id': str(text_id), 'text': text, 'mention_data': mentions_optim})
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


def split_eval_mention(num):
    dev_mention_data = com_utils.pickle_load(fileConfig.dir_ner + fileConfig.file_ner_eval_mention_data)
    data_len = len(dev_mention_data)
    block_size = data_len / num
    for i in range(1, num + 1):
        data_iter = dev_mention_data[int((i - 1) * block_size): int(i * block_size)]
        com_utils.pickle_save(data_iter, fileConfig.dir_ner_split + fileConfig.file_ner_eval_mention_split.format(i))
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
    alia_kb_df.fillna('')
    count = 0
    for dev_data in tqdm(dev_mention_data, desc='find {} cands'.format(index)):
        # count += 1
        # if (count < 465):
        #     continue
        mention_data = dev_data['mention_data']
        for mention in mention_data:
            mention_text = mention['mention']
            if mention_text is None:
                continue
            cands = []
            cand_ids = {}
            # match orginal
            mention_text_proc = com_utils.cht_to_chs(mention_text.lower())
            mention_text_proc = com_utils.complete_brankets(mention_text_proc)
            # print(mention_text_proc)
            mention_text_proc_extend = mention_text_proc[0:len(mention_text_proc) - 1]
            subject_df = data_utils.pandas_query(pd_df, 'subject', mention_text_proc)
            for _, item in subject_df.iterrows():
                s_id = str(item['subject_id'])
                if cand_ids.get(s_id) is not None:
                    continue
                cand_ids[s_id] = 1
                subject = item['subject']
                # text = data_utils.get_text(ast.literal_eval(item['data']), item['subject'])
                text = data_utils.get_all_text(item['subject'], ast.literal_eval(item['data']))
                cands.append({'cand_id': s_id, 'cand_subject': subject, 'cand_text': text,
                              'cand_type': com_utils.get_kb_type(ast.literal_eval(item['type']))})
            # match more
            # subject_df = data_utils.pandas_query(pd_df, 'subject', mention_text_proc_extend)
            # for _, item in subject_df.iterrows():
            #     s_id = str(item['subject_id'])
            #     if cand_ids.get(s_id) is not None:
            #         continue
            #     cand_ids[s_id] = 1
            #     subject = item['subject']
            #     # text = data_utils.get_text(ast.literal_eval(item['data']), item['subject'])
            #     text = data_utils.get_all_text(item['subject'], ast.literal_eval(item['data']))
            #     cands.append({'cand_id': s_id, 'cand_subject': subject, 'cand_text': text,
            #                   'cand_type': com_utils.get_kb_type(ast.literal_eval(item['type']))})
            # match alias
            alias_subject_ids = []
            # match orginal
            alias_df = data_utils.pandas_query(alia_kb_df, 'subject', mention_text_proc)
            for _, item in alias_df.iterrows():
                a_id = str(item['subject_id'])
                if alias_subject_ids.__contains__(a_id):
                    continue
                alias_subject_ids.append(a_id)
            # match more
            # alias_df = data_utils.pandas_query(alia_kb_df, 'subject', mention_text_proc_extend)
            # for _, item in alias_df.iterrows():
            #     a_id = str(item['subject_id'])
            #     if alias_subject_ids.__contains__(a_id):
            #         continue
            #     alias_subject_ids.append(a_id)
            for alia_id in alias_subject_ids:
                alias_df = pd_df[pd_df['subject_id'] == int(alia_id)]
                for _, item in alias_df.iterrows():
                    b_id = str(item['subject_id'])
                    if cand_ids.get(b_id) is not None:
                        continue
                    cand_ids[b_id] = 1
                    subject = item['subject']
                    # text = data_utils.get_text(ast.literal_eval(item['data']), item['subject'])
                    text = data_utils.get_all_text(item['subject'], ast.literal_eval(item['data']))
                    cands.append({'cand_id': b_id, 'cand_subject': subject, 'cand_text': text,
                                  'cand_type': com_utils.get_kb_type(ast.literal_eval(item['type']))})
            # match gen subject
            # gen_subject_ids = []
            # for gen_subject in mention['gen_subjects']:
            #     gen_text = com_utils.cht_to_chs(gen_subject.lower())
            #     alias_df = alia_kb_df[alia_kb_df['subject'] == gen_text]
            #     for _, item in alias_df.iterrows():
            #         a_id = str(item['subject_id'])
            #         if gen_subject_ids.__contains__(a_id):
            #             continue
            #         gen_subject_ids.append(a_id)
            #     for alia_id in gen_subject_ids:
            #         alias_df = pd_df[pd_df['subject_id'] == int(alia_id)]
            #         for _, item in alias_df.iterrows():
            #             b_id = str(item['subject_id'])
            #             if cand_ids.get(b_id) is not None:
            #                 continue
            #             cand_ids[b_id] = 1
            #             subject = item['subject']
            #             # text = data_utils.get_text(ast.literal_eval(item['data']), item['subject'])
            #             text = data_utils.get_all_text(item['subject'], ast.literal_eval(item['data']))
            #             cands.append({'cand_id': b_id, 'cand_subject': subject, 'cand_text': text,
            #                           'cand_type': com_utils.get_kb_type(ast.literal_eval(item['type']))})
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


def create_eval_cands_from_split(num):
    out_file = open(fileConfig.dir_ner + fileConfig.file_ner_eval_cands_data, 'w', encoding='utf-8')
    for i in range(1, num + 1):
        datas = com_utils.pickle_load(fileConfig.dir_ner_split + fileConfig.file_ner_eval_cands_split.format(i))
        for line in datas:
            text = ujson.dumps(line, ensure_ascii=False)
            out_file.write(text)
            out_file.write('\n')
    print("merge eval cands data success!")


def create_test_data():
    print("start create test data")
    split_num = 8
    mode = 2
    create_mention_data(mode)
    # gen_simi_subject_list(fileConfig.dir_ner + fileConfig.file_ner_test_mention_data)
    split_test_mention(split_num)
    create_dev_mention_cands_multi(mention_file_name=fileConfig.file_ner_test_mention_split,
                                   out_file_name=fileConfig.file_ner_test_cands_split, num=split_num)
    create_test_cands_from_split(split_num)
    print("success create fasttext model test data")


def create_eval_data():
    print("start create eval data")
    split_num = 8
    mode = 3
    create_mention_data(mode)
    # gen_simi_subject_list(fileConfig.dir_ner + fileConfig.file_ner_test_mention_data)
    split_eval_mention(split_num)
    create_dev_mention_cands_multi(mention_file_name=fileConfig.file_ner_eval_mention_split,
                                   out_file_name=fileConfig.file_ner_eval_cands_split, num=split_num)
    create_eval_cands_from_split(split_num)
    print("success create fasttext model test data")


def create_predict_data():
    print("start create predict data")
    split_num = 8
    mode = 1
    create_mention_data(mode)
    split_dev_mention(split_num)
    create_dev_mention_cands_multi(mention_file_name=fileConfig.file_ner_dev_mention_split,
                                   out_file_name=fileConfig.file_ner_dev_cands_split, num=split_num)
    create_dev_cands_from_split(split_num)
    print("success create fasttext model predict data")


def debug_method():
    index = 1
    mention_file = fileConfig.dir_ner_split + fileConfig.file_ner_test_mention_split.format(index)
    pd_file = fileConfig.dir_kb_split + fileConfig.file_kb_pandas_split.format(index)
    alia_kb_file = fileConfig.dir_kb_info + fileConfig.file_kb_pandas_alias_data
    out_file = fileConfig.dir_ner_split + fileConfig.file_ner_test_cands_split.format(index)
    create_dev_mention_cands_data(index, mention_file, pd_file, alia_kb_file, out_file)


if __name__ == '__main__':
    if len(sys.argv) == 1 or not sys.argv[1] in ['test', 'predict', 'debug', 'eval']:
        print("should input param [test/predict/debug]")
    if sys.argv[1] == 'test':
        create_test_data()
    elif sys.argv[1] == 'predict':
        create_predict_data()
    elif sys.argv[1] == 'debug':
        debug_method()
    elif sys.argv[1] == 'eval':
        create_eval_data()
