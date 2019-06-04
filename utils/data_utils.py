import config
import torch
import random
import os
import string
import ast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from entity.data import NERProcessor
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import com_utils
from entity.data import DataSet, NELToTensor

# init params
comConfig = config.ComConfig()
nerConfig = config.NERConfig()
nelConfig = config.NELConfig()
fileConfig = config.FileConfig()
fasttextConfig = config.FastTextConfig()


def init_params():
    processors = {nerConfig.ner_task_name: NERProcessor}
    task_name = nerConfig.ner_task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    tokenizer = BertTokenizer(vocab_file=fileConfig.dir_bert + fileConfig.file_bert_vocab)
    return processor, tokenizer


def create_ner_batch_iter(mode):
    processor, tokenizer = init_params()
    if mode == nerConfig.mode_train:
        examples = processor.get_train_examples()
        num_train_steps = int(len(
            examples) / nerConfig.train_batch_size / nerConfig.gradient_accumulation_steps * nerConfig.num_train_epochs)
        batch_size = nerConfig.train_batch_size
        print("{} Num steps = {}".format(mode, num_train_steps))
    elif mode == nerConfig.mode_dev:
        examples = processor.get_dev_examples()
        batch_size = nerConfig.eval_batch_size
    elif mode == nerConfig.mode_predict:
        examples = processor.get_predict_examples()
        batch_size = nerConfig.eval_batch_size
    elif mode == nerConfig.mode_test:
        examples = processor.get_dev_examples()
        batch_size = nerConfig.eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()
    # 特征
    features = processor.convert_examples_to_features(examples, label_list, nerConfig.max_seq_length, tokenizer)
    print("{} Num examples = {}".format(mode, len(examples)))
    print("{} Batch size = {}".format(mode, batch_size))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)
    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)
    if mode == nerConfig.mode_train:
        sampler = RandomSampler(data)
    elif mode == nerConfig.mode_dev:
        sampler = SequentialSampler(data)
    elif mode == nerConfig.mode_predict:
        sampler = SequentialSampler(data)
    elif mode == nerConfig.mode_test:
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)
    # 迭代器
    iterator = DataLoader(data, num_workers=4, sampler=sampler, batch_size=batch_size)
    if mode == nerConfig.mode_train:
        return iterator, num_train_steps
    elif mode == nerConfig.mode_dev:
        return iterator
    elif mode == nerConfig.mode_predict:
        return iterator, examples
    elif mode == nerConfig.mode_test:
        return iterator, examples
    else:
        raise ValueError("Invalid mode %s" % mode)


def deal_ner_predict_data(predict_list, data_list, out_file):
    # init label map
    id2label = {i: label for i, label in enumerate(nerConfig.labels)}
    # init predict list
    label_list = []
    for labels in predict_list:
        for label in labels:
            item = label[label != -1]
            label_list.append(item)
    datas = []
    for data, label in zip(data_list, label_list):
        text_list = data.text_a
        assert len(text_list) == len(label)
        labels = []
        for item in label:
            labels.append(id2label.get(item.item()))
        datas.append({'text': text_list, 'tag': labels, 'mention_data_original': data.mention_data})
    com_utils.pickle_save(datas, out_file)


def get_entity_cands(kb_entity, kb_id, pd_df):
    cands = []
    entity_ids = []
    txts = []
    cands.append(kb_entity['subject'])
    entity_ids.append(kb_id)
    txts.append(kb_entity['text'])
    # start = com_utils.get_time()
    # df = pd_df[pd_df['subject'] == kb_entity['subject']]
    # if len(df) > 0:
    #     for _, item in df.iterrows():
    #         id = str(item['subject_id'])
    #         if id in entity_ids:
    #             continue
    #         cands.append(item['subject'])
    #         entity_ids.append(id)
    #         txts.append(get_text(ast.literal_eval(item['data']), item['subject']))
    #         if len(cands) == nelConfig.max_cands_num:
    #             break
    # 先不用alias的数据
    # alias = kb_entity['alias']
    # if len(alias) > 0:
    #     for alia in alias:
    #         df = pd_df[pd_df['subject'] == alia]
    #         if len(df) > 0:
    #             for _, item in df.iterrows():
    #                 id = str(item['subject_id'])
    #                 if id in entity_ids:
    #                     continue
    #                 cands.append(item['subject'])
    #                 entity_ids.append(id)
    #                 txts.append(get_text(ast.literal_eval(item['data']), item['subject']))
    return cands, entity_ids, txts


def reorder_sequence(emb_sequence, order):
    order = torch.LongTensor(order).to(comConfig.device)
    return emb_sequence.index_select(index=order, dim=0)


def reorder_lstm_states(states, order):
    assert isinstance(states, tuple)
    assert len(states) == 2
    assert states[0].size() == states[1].size()
    assert len(order) == states[0].size()[1]

    order = torch.LongTensor(order).to(comConfig.device)
    sorted_states = (states[0].index_select(index=order, dim=1), states[1].index_select(index=order, dim=1))
    return sorted_states


def get_mask(lens):
    '''
    :param lens: list of batch, every item is a int means the length of a sample
    :return: [batch, max_seq_len]
    '''
    max_len = max(lens)
    batch_size = len(lens)
    seq_range = torch.arange(max_len).long().to(comConfig.device)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)

    seq_length = lens.unsqueeze(1).expand(batch_size, max_len)
    mask = seq_range < seq_length
    return mask.float()


def get_final_encoder_states(encoder_outputs, mask, bidirectional=False):
    last_word_indices = mask.sum(1).long - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2)]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def create_nel_batch_iter(mode, entity_context_vocab,
                          mention_context_vocab, entity_vocab):
    if mode == nelConfig.mode_train:
        datas = com_utils.pickle_load(fileConfig.dir_nel + fileConfig.file_nel_entity_link_train_data)
        data_len = len(datas)
        train_size = int(data_len * comConfig.train_ratio)
        random.seed(comConfig.random_seed)
        random.shuffle(datas)
        train_data = datas[:train_size]
        dev_data = datas[train_size:]
        print("create nel data train len:{} dev len:{}".format(len(train_data), len(dev_data)))
        train_dataset = DataSet(train_data, transform=transforms.Compose(
            [NELToTensor(entity_context_vocab, mention_context_vocab, entity_vocab)]))
        dev_dataset = DataSet(dev_data, transform=transforms.Compose(
            [NELToTensor(entity_context_vocab, mention_context_vocab, entity_vocab)]))
        train_iter = DataLoader(train_dataset, num_workers=4, batch_size=nelConfig.batch_size, shuffle=True,
                                collate_fn=collate_fn_entity_link)
        dev_iter = DataLoader(dev_dataset, num_workers=4, batch_size=nelConfig.batch_size, shuffle=True,
                              collate_fn=collate_fn_entity_link)
        return train_iter, dev_iter


def collate_fn_entity_link(batches):
    '''
    'mention_context, '
    'mention_position, '
    'entity_cands_id, '
    'entity_contexts_id'
    '''
    mention_context_max_len = 0
    entity_contexts_max_len = 0
    entity_cands_max_len = 0
    entity_contexts_list_max_len = 0
    for batch in batches:
        mention_context_max_len = max(len(batch['mention_context']), mention_context_max_len)
        entity_contexts_max_len = max(len(max(batch['entity_contexts_id'], key=len)), entity_contexts_max_len)
        if len(batch['entity_contexts_id']) > entity_contexts_list_max_len:
            entity_contexts_list_max_len = len(batch['entity_contexts_id'])
        if len(batch['entity_cands_id']) > entity_cands_max_len:
            entity_cands_max_len = len(batch['entity_cands_id'])
    mention_context = []
    entity_context = []
    pos = []
    cand_id = []
    for batch in batches:
        # 复制数据到最大长度
        entity_context_len_gap = entity_contexts_list_max_len - len(batch['entity_contexts_id'])
        batch['entity_contexts_id'] = batch['entity_contexts_id'] + [
            batch['entity_contexts_id'][0]] * entity_context_len_gap
        mention_context.append(torch.LongTensor(
            batch['mention_context'] + [0] * (mention_context_max_len - len(batch['mention_context']))))
        p = [context + [0] * (entity_contexts_max_len - len(context)) for context in batch['entity_contexts_id']]
        # if len(p) < entity_contexts_list_max_len:
        #     length = entity_contexts_list_max_len - len(p)
        #     for _ in range(length):
        #         p.append([0] * entity_contexts_max_len)
        entity_context.append(torch.LongTensor(p))
        pos.append(com_utils.conv_list_to_int(batch['mention_position']))
        # 复制数据到最大长度
        cand_id.append(torch.LongTensor(batch['entity_cands_id'] + [batch['entity_cands_id'][0]] * (
                entity_cands_max_len - len(batch['entity_cands_id']))))
        # cand_id.append(torch.LongTensor(batch['entity_cands_id']))
    if 'target_position' in batches[0]:
        target = [batch['target_position'] for batch in batches]
        return {'mention_context': torch.stack(mention_context, dim=0),
                'mention_position': torch.LongTensor(pos),
                'entity_cands_id': torch.stack(cand_id, dim=0),
                'entity_contexts_id': torch.stack(entity_context, dim=0),
                'target': torch.LongTensor(target)
                }
    else:
        mentions = [batch['mention'] for batch in batches]
        text_ids = [batch['text_id'] for batch in batches]
        return {
            'mention_context': torch.stack(mention_context, dim=0),
            'mention_position': torch.LongTensor(pos),
            'entity_cands_id': torch.stack(cand_id, dim=0),
            'entity_contexts_id': torch.stack(entity_context, dim=0),
            'mention': mentions,
            'text_id': text_ids
        }


def get_stopword_list(file_path):
    assert os.path.exists(file_path)
    stopword_list = {}
    for word in open(file_path):
        text = word.strip('\n')
        if stopword_list.get(text) is None:
            stopword_list[text] = 1
        else:
            stopword_list[text] += 1
    return stopword_list


def get_text(data, subject):
    if len(data) == 0:
        return subject
    for i in range(len(data)):
        if data[i]['predicate'] == '摘要':
            return data[i]['object']
    for i in range(len(data)):
        if data[i]['predicate'] == '义项描述':
            return data[i]['object']
    max_len = 0
    max_text = ''
    # 找最长的描述
    for i in range(len(data)):
        if len(data[i]['predicate']) > max_len:
            max_len = len(data[i]['predicate'])
            max_text = data[i]['object']
    return max_text


def get_kb_text_list(kb_str):
    result_texts = []
    subject_list = set()
    original_subject = kb_str['subject'].strip()
    subject_list.add(kb_str['subject'].strip())
    alias = kb_str['alias']
    for alia in alias:
        subject_list.add(alia.strip())
    text = get_text(kb_str['data'], kb_str['subject'])
    # find original_subject
    if text.find(original_subject) == -1:
        for subject in subject_list:
            if text.find(subject) > -1:
                original_subject = subject
                break
    for i, subject in enumerate(subject_list):
        if i == fasttextConfig.max_alias_num:
            break
        text_change = text.replace(original_subject, subject)
        result_texts.append(text_change)
    return result_texts


def get_jieba_split_words(text, jieba, stopwords):
    texts = jieba.lcut(text)
    result_list = []
    for text in texts:
        if stopwords.get(text) is None:
            result_list.append(text)
    return result_list


def strip_punctuation(text):
    result = text
    for c in string.punctuation:
        result = result.strip(c)
        result = result.replace(c, '_')
    return result
