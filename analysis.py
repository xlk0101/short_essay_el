import ujson
import config
import torch
import os

from collections import Counter
from utils.zh_wiki import zh2Hans
from utils import com_utils, data_utils, text_cut
from tqdm import tqdm
import fastText
import re

# init params
fasttextConfig = config.FastTextConfig()
fileConfig = config.FileConfig()
comConfig = config.ComConfig()
nerConfig = config.NERConfig()
cut_client = text_cut.get_client()


def anlalysis_data_set():
    # train_file = open(fileConfig.dir_data + fileConfig.file_extend_train_data, 'r')
    # test_datas = com_utils.pickle_load(fileConfig.dir_data + fileConfig.file_test_pkl)
    dev_file = open(fileConfig.dir_data + fileConfig.file_eval_data)
    text_list = []
    max_length = 0
    text_len_dict = {}
    max_text = ''
    # for line in tqdm(train_file, 'deal train file'):
    #     jstr = ujson.loads(line)
    #     text = jstr['text'].strip()
    #     text_len = len(text)
    #     if not text_len_dict.get(text_len):
    #         text_len_dict[text_len] = 1
    #     else:
    #         text_len_dict[text_len] += 1
    #     if text_len > max_length:
    #         max_length = text_len
    #         max_text = text
    #     text_list.append(text)
    # for line in tqdm(test_datas, 'deal test file'):
    #     jstr = ujson.loads(line)
    #     text = jstr['text'].strip()
    #     text_len = len(text)
    #     if not text_len_dict.get(text_len):
    #         text_len_dict[text_len] = 1
    #     else:
    #         text_len_dict[text_len] += 1
    #     if text_len > max_length:
    #         max_length = text_len
    #         max_text = text
    #     text_list.append(text)
    for line in tqdm(dev_file, 'deal dev file'):
        jstr = ujson.loads(line)
        text = jstr['text'].strip()
        text_len = len(text)
        if not text_len_dict.get(text_len):
            text_len_dict[text_len] = 1
        else:
            text_len_dict[text_len] += 1
        if text_len > max_length:
            max_length = text_len
            max_text = text
        text_list.append(text)
    data_len = len(text_list)
    print("the data set length is {}".format(data_len))
    print("the max text length is {}".format(max_length))
    print("the max text is {}".format(max_text))
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
    # tensor = torch.LongTensor([0] * 30).to(torch.device('cuda'))
    # print(tensor)
    model = fastText.load_model(
        fileConfig.dir_fasttext + fileConfig.file_fasttext_model.format(fasttextConfig.choose_model))
    print("test")


def analysis_train_mentions():
    train_file = open(fileConfig.dir_data + fileConfig.file_train_data, 'r', encoding='utf-8')
    count = 1
    for line in tqdm(train_file, 'deal train file'):
        jstr = ujson.loads(line)
        text = jstr['text']
        mention_data = jstr['mention_data']
        for mention in mention_data:
            if text.find(mention['mention']) == -1:
                print('item:{} mention:{} text:{}'.format(count, mention['mention'], text))
                count += 1


def get_result_error_list(gen_mentions, original_mentions, gen_more_dict):
    result_list = []
    gen_indexs = [0] * len(gen_mentions)
    original_indexs = [0] * len(original_mentions)
    # Traverse gen mentions
    for i, gen_mention in enumerate(gen_mentions):
        for j, original_mention in enumerate(original_mentions):
            if gen_mention['offset'] == original_mention['offset'] and gen_mention['kb_id'] == original_mention[
                'kb_id']:
                gen_indexs[i] = 1
                original_indexs[j] = 1
                continue
            elif gen_mention['offset'] == original_mention['offset'] and gen_mention['kb_id'] != original_mention[
                'kb_id']:
                gen_indexs[i] = 1
                original_indexs[j] = 1
                result_list.append({'error_type': comConfig.result_error_type_miss, 'gen_mention': gen_mention,
                                    'original_mention': original_mention})
                continue
        if gen_indexs[i] == 0:
            gen_indexs[i] = 1
            result_list.append({'error_type': comConfig.result_error_type_gen_more, 'gen_mention': gen_mention})
            if len(gen_mention['mention']) > 1:
                com_utils.dict_add(gen_more_dict, gen_mention['mention'])
    # Traverse original mentions
    for i, value in enumerate(original_indexs):
        if value == 0 and original_mentions[i]['kb_id'] != 'NIL':
            result_list.append(
                {'error_type': comConfig.result_error_type_original_more, 'original_mention': original_mentions[i]})
    return result_list


def get_error_type(type_index):
    if type_index == comConfig.result_error_type_miss:
        return 'mention miss match'
    elif type_index == comConfig.result_error_type_gen_more:
        return 'generate more data'
    elif type_index == comConfig.result_error_type_original_more:
        return 'not find original data'


def get_error_content(kb_dict, error_item):
    result_str = ''
    if error_item['error_type'] == comConfig.result_error_type_miss:
        result_str += 'gen_mention:'
        result_str += str(error_item['gen_mention']) + '\n'
        result_str += 'gen_kb_entity:'
        gen_kb_entity = kb_dict.get(error_item['gen_mention']['kb_id'])
        if gen_kb_entity is not None:
            result_str += str(gen_kb_entity)[0:200]
        else:
            result_str += 'NIL'
        result_str += '\n'
        result_str += 'original_mention:'
        result_str += str(error_item['original_mention']) + '\n'
        result_str += 'original_kb_entity:'
        original_kb_entity = kb_dict.get(error_item['original_mention']['kb_id'])
        if original_kb_entity is not None:
            result_str += str(original_kb_entity)[0:200]
        else:
            result_str += 'NIL'
        result_str += '\n'
    elif error_item['error_type'] == comConfig.result_error_type_gen_more:
        result_str += 'gen_mention:'
        result_str += str(error_item['gen_mention']) + '\n'
        result_str += 'gen_kb_entity:'
        gen_kb_entity = kb_dict.get(error_item['gen_mention']['kb_id'])
        if gen_kb_entity is not None:
            result_str += str(gen_kb_entity)[0:200]
        else:
            result_str += 'NIL'
        result_str += '\n'
    elif error_item['error_type'] == comConfig.result_error_type_original_more:
        result_str += 'original_mention:'
        result_str += str(error_item['original_mention']) + '\n'
        result_str += 'original_kb_entity:'
        original_kb_entity = kb_dict.get(error_item['original_mention']['kb_id'])
        if original_kb_entity is not None:
            result_str += str(original_kb_entity)[0:200]
        else:
            result_str += 'NIL'
        result_str += '\n'
    return result_str


def analysis_test_result():
    print("start analysis test result...")
    test_result_file = open(fileConfig.dir_result + fileConfig.file_result_fasttext_test, 'r', encoding='utf-8')
    kb_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_kb_dict)
    error_results = []
    more_dict = {}
    for line in tqdm(test_result_file, 'read from file'):
        jstr = ujson.loads(line)
        gen_mentions = jstr['mention_data']
        original_mentions = jstr['mention_data_original']
        error_list = get_result_error_list(gen_mentions, original_mentions, more_dict)
        if (len(error_list)) > 0:
            error_results.append({'text_id': jstr['text_id'], 'text': jstr['text'], 'errors': error_list})
    test_result_file.close()
    test_result_file = None
    out_file = open(fileConfig.dir_result + fileConfig.file_result_fasttext_test_analysis, 'w', encoding='utf-8')
    for item in tqdm(error_results, 'write result'):
        out_file.write('-' * 20)
        out_file.write('\n')
        out_file.write("text_id:{}--text:{}".format(item['text_id'], item['text']))
        out_file.write('\n')
        out_file.write("errors:")
        out_file.write('\n')
        for error in item['errors']:
            out_file.write('*' * 20)
            out_file.write('\n')
            out_file.write('error type:{}'.format(get_error_type(error['error_type'])))
            out_file.write('\n')
            out_file.write(get_error_content(kb_dict, error))
            out_file.write('*' * 20)
            out_file.write('\n')
        out_file.write('-' * 20)
        out_file.write('\n')
    out_more = open(fileConfig.dir_result + fileConfig.file_analysis_gen_more, 'w', encoding='utf-8')
    for item in tqdm(Counter(more_dict).most_common(), 'write more'):
        out_more.write(item[0])
        out_more.write('\n')


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


def analysis_test_ner_result():
    print("start analysis test ner result...")
    ner_test_datas = com_utils.pickle_load(fileConfig.dir_ner + fileConfig.file_ner_test_predict_tag)
    jieba_dict = com_utils.pickle_load(fileConfig.dir_kb_info + fileConfig.file_jieba_kb)
    out_file = open(fileConfig.dir_result + fileConfig.file_ner_test_result_analysis, 'w', encoding='utf-8')
    stopwords = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_stopword)
    gen_more_words = data_utils.get_stopword_list(fileConfig.dir_stopword + fileConfig.file_analysis_gen_more)
    text_id = 1
    for data in tqdm(ner_test_datas, 'find entity'):
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
                mention_length += 1
                is_find = True
            elif tag.find(nerConfig.E_seg) > -1 and not is_find:
                type_str = tag.split('_')[1]
                type_dict = com_utils.dict_add(type_dict, type_str)
                start_index = i
                mention_length += 1
                mention = text[start_index:start_index + mention_length]
                # mention = data_utils.strip_punctuation(mention)
                type_list = Counter(type_dict).most_common()
                mentions.append({'T': 'NER', 'mention': mention, 'offset': str(start_index), 'type': type_list[0][0]})
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
                # mention = data_utils.strip_punctuation(mention)
                type_list = Counter(type_dict).most_common()
                mentions.append({'T': 'NER', 'mention': mention, 'offset': str(start_index), 'type': type_list[0][0]})
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
            # if tag.find(nerConfig.B_seg) > -1 or tag.find(nerConfig.I_seg) > -1 or tag.find(nerConfig.E_seg) > -1:
            jieba_offset = i
            jieba_char = text[i]
            jieba_text = get_jieba_mention(jieba_entities, jieba_char)
            if jieba_text is None:
                continue
            elif jieba_text == '_' or jieba_text == '-':
                continue
            elif len(jieba_text) == 1:
                continue
            elif stopwords.get(jieba_text) is not None:
                continue
            elif gen_more_words.get(jieba_text) is not None:
                continue
            jieba_offset = jieba_offset - jieba_text.find(jieba_char)
            if len(jieba_text) <= comConfig.max_jieba_cut_len and (
                    jieba_dict.get(jieba_text) is not None):
                type_str = tag.split('_')[1] if tag.find('_') > -1 else 'O'
                if jieba_text is None:
                    continue
                if not is_already_find_mention(mentions, jieba_text, jieba_offset):
                    # jieba_offset = text.find(jieba_text)
                    mentions.append(
                        {'T': 'JIEBA', 'mention': jieba_text, 'offset': str(jieba_offset), 'type': type_str})
        # find inner brackets mentions
        bracket_mentions = data_utils.get_mention_inner_brackets(text, tag_list)
        for mention in bracket_mentions:
            mention['T'] = 'bracket'
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
        #                     {'T': 'COM', 'mention': mention['mention'], 'offset': str(find_offset),
        #                      'type': mention['type']})
        #     except BaseException:
        #         # print("occur error when match mention str in completion mentions, error value:{} text:{}".format(
        #         #     mention_str, text))
        #         pass
        #     mentions_com.append(mention)
        # mentions = mentions_com
        # completion mentions
        out_file.write('\n')
        result_str = ''
        for i in range(len(text)):
            result_str += text[i] + '-' + tag_list[i] + ' '
        out_file.write(' text_id:{}, text:{} '.format(text_id, result_str))
        out_file.write('\n')
        out_file.write(' gen_mentions:{} '.format(ujson.dumps(mentions, ensure_ascii=False)))
        out_file.write('\n')
        text_id += 1


def change_gen_more_word():
    gen_more_file = open(fileConfig.dir_stopword + fileConfig.file_analysis_gen_more, 'r', encoding='utf-8')
    datas = gen_more_file.readlines()
    infos = []
    for item in datas:
        text = item.strip('\n')
        if len(text) <= 1 or len(text) > 3:
            continue
        if not infos.__contains__(text):
            infos.append(text)
    write_file = open(fileConfig.dir_stopword + fileConfig.file_analysis_gen_more, 'w', encoding='utf-8')
    for info in infos:
        write_file.write(info)
        write_file.write('\n')
    print('success change gen more file')


if __name__ == '__main__':
    anlalysis_data_set()
    # anlalysis_miss_text()
    # test()
    # analysis_train_data()
    # analysis_kb_type()
    # analysis_train_mentions()
    # analysis_test_result()
    # analysis_test_ner_result()
    # change_gen_more_word()
