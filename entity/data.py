import ujson
import config

from tqdm import tqdm
from utils import com_utils
from entity.entity import InputExample, InputFeature
from collections import Counter
from typing import List

# init parmas
fileConfig = config.FileConfig()
nerConfig = config.NERConfig()


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""

    def get_train_examples(self, data_dir):
        """读取训练集 Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()


class NERProcessor(DataProcessor):
    """将数据构造成example格式"""

    def _create_example(self, lines, set_type):
        examples = []
        for line in lines:
            guid = "%s-%s" % (set_type, line['id'])
            text_a = line["text"]
            label = line["tag"]
            mention_data = []
            if set_type == nerConfig.mode_dev or set_type == nerConfig.mode_train:
                mention_data = line['mention_data']
            assert len(label) == len(text_a)
            example = InputExample(guid=guid, text_a=text_a, label=label, mention_data=mention_data)
            examples.append(example)
        return examples

    def get_train_examples(self):
        lines = com_utils.pickle_load(fileConfig.dir_ner
                                      + fileConfig.file_ner_train_data)
        examples = self._create_example(lines, nerConfig.mode_train)
        return examples

    def get_dev_examples(self):
        lines = com_utils.pickle_load(fileConfig.dir_ner
                                      + fileConfig.file_ner_dev_data)
        examples = self._create_example(lines, nerConfig.mode_dev)
        return examples

    def get_predict_examples(self):
        lines = com_utils.pickle_load(fileConfig.dir_ner + fileConfig.file_ner_predict_data)
        examples = self._create_example(lines, nerConfig.mode_predict)
        return examples

    def get_labels(self):
        return nerConfig.labels

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        # 标签转换为数字
        label_map = {label: i for i, label in enumerate(label_list)}
        # load sub_vocab
        sub_vocab = {}
        vocab = {}
        with open(fileConfig.dir_bert + fileConfig.file_bert_vocab, 'r') as fr:
            for line in fr:
                _line = line.strip('\n')
                if vocab.get(_line) is None:
                    vocab[_line] = 1
                else:
                    vocab[_line] += 1
                if "##" in _line and sub_vocab.get(_line) is None:
                    sub_vocab[_line] = 1
        features = []
        for ex_index, example in tqdm(enumerate(examples), desc='example to feature'):
            # tokens_a = tokenizer.tokenize(example.text_a)
            tokens_a = com_utils.replace_useless_item(example.text_a, vocab)
            labels = example.label
            if len(tokens_a) != len(labels):
                print("token and label can't match")
            if len(tokens_a) == 0 or len(labels) == 0:
                continue
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                labels = labels[:(max_seq_length - 2)]
            # ----------------处理source--------------
            ## 句子首尾加入标示符
            tokens = [nerConfig.CLS_flag] + tokens_a + [nerConfig.SEP_flag]
            # 保存input_length
            # input_len = len(tokens)
            input_len = max_seq_length
            segment_ids = [0] * len(tokens)
            ## 词转换成数字
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # ---------------处理target----------------
            ## Notes: label_id中不包括[CLS]和[SEP]
            label_id = [label_map[l] for l in labels]
            label_id = label_id
            label_padding = [-1] * (max_seq_length - len(label_id))
            label_id += label_padding
            ## output_mask用来过滤bert输出中sub_word的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
            ## 此外，也是为了适应crf
            output_mask = [0 if sub_vocab.get(t) is not None else 1 for t in tokens_a]
            output_mask = [0] + output_mask + [0]
            output_mask += padding
            # ----------------处理后结果-------------------------
            # for example, in the case of max_seq_length=10:
            # raw_data:          春 秋 忽 代 谢le
            # token:       [CLS] 春 秋 忽 代 谢 ##le [SEP]
            # input_ids:     101 2  12 13 16 14 15   102   0 0 0
            # input_mask:      1 1  1  1  1  1   1     1   0 0 0
            # label_id:          T  T  O  O  O
            # output_mask:     0 1  1  1  1  1   0     0   0 0 0
            # --------------看结果是否合理------------------------
            if ex_index < 1:
                print("-----------------Example-----------------")
                print("guid: %s" % (example.guid))
                print("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print("label: %s " % " ".join([str(x) for x in label_id]))
                print("output_mask: %s " % " ".join([str(x) for x in output_mask]))
                print("input len:{}".format(input_len))
            # ----------------------------------------------------

            feature = InputFeature(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                   label_id=label_id, output_mask=output_mask, input_length=input_len)
            features.append(feature)
        return features


class DataSet():
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.transform(sample)
        return sample


class NELToTensor():
    def __init__(self, entity_context_vocab, mention_context_vocab, entity_vocab):
        self.entity_context_vocab = entity_context_vocab
        self.mention_context_vocab = mention_context_vocab
        self.entity_vocab = entity_vocab

    def __call__(self, sample):
        mention: List = list(sample['mention'])
        mention_context: List = list(sample['mention_text'])
        mention_position: List = sample['mention_position']
        # entity_cands: List = sample['entity_cands']
        entity_cands: List = sample['entity_ids']
        entity_contexts: List[List[str]] = [list(context) for context in sample['entity_text']]
        target_id = sample['entity_ids'][0]
        target_position = entity_cands.index(target_id)

        mention_id = [self.mention_context_vocab.word2id(m) for m in mention]
        mention_context_id = [self.mention_context_vocab.word2id(m) for m in mention_context]
        entity_cands_id = [self.entity_vocab.word2id(m, type='label') for m in entity_cands]
        entity_contexts_id = [[self.entity_context_vocab.word2id(m[i]) for i in range(len(m))] for m in entity_contexts]

        return {'mention': mention_id,
                'mention_context': mention_context_id,
                'mention_position': mention_position,
                'entity_cands_id': entity_cands_id,
                'entity_contexts_id': entity_contexts_id,
                'target_id': target_id,
                'target_position': target_position}
