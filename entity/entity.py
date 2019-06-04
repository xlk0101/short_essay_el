class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask = output_mask


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, mention_data=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.mention_data = mention_data


class Vocab():
    def __init__(self, vocab_file):
        self.word_to_idx = {}
        self.id_to_word = {}
        with open(vocab_file, 'r') as fr:
            for line in fr.readlines():
                word = line.split('\n')[0]
                if word in self.word_to_idx:
                    raise ("Duplicate word : {} exist".format(word))
                self.word_to_idx[word] = len(self.word_to_idx)
                self.id_to_word[len(self.word_to_idx) - 1] = word

    def word2id(self, word, type='token'):
        if type == 'token':
            return self.word_to_idx.get(word, self.word_to_idx['[UNK]'])
        else:
            return self.word_to_idx[word]

    def id2word(self, id):
        return self.id_to_word[id]

    def __len__(self):
        return len(self.word_to_idx)
