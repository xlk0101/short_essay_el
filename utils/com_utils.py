import os
import pickle
import config
import time
import torch
from datetime import timedelta
from utils.zh_wiki import zh2Hans
from model.ner_bert_crf import Ner_Bert_Crf

fileConfig = config.FileConfig()
nerConfig = config.NERConfig()


class ProgressBar(object):
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, epoch_size, batch_size, max_arrow=20):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.max_steps = round(epoch_size / batch_size)  # 总共处理次数 = round(epoch/batch_size)
        self.max_arrow = max_arrow  # 进度条的长度

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, train_acc, train_loss, f1, used_time, i):
        num_arrow = int(i * self.max_arrow / self.max_steps)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        num_steps = self.batch_size * i  # 当前处理数据条数
        process_bar = '%d' % num_steps + '/' + '%d' % self.epoch_size + '[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%' + ' - train_acc ' + '%.4f' % train_acc + ' - train_loss ' + \
                      '%.4f' % train_loss + ' - f1 ' + '%.4f' % f1 + ' - time ' + '%s' % timedelta(
            seconds=used_time)
        print('\r' + process_bar, end='')  # 这两句打印字符到终端


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def pickle_save(items, path):
    pickle.dump(items, open(path, 'wb'))


def pickle_load(path):
    return pickle.load(open(path, 'rb'))


def save_model(model, output_dir):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, fileConfig.file_ner_model)
    torch.save(model_to_save.state_dict(), output_model_file)


def load_model(output_dir, num_tag):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(output_dir, fileConfig.file_ner_model)
    model_state_dict = torch.load(output_model_file)
    model = Ner_Bert_Crf.from_pretrained(fileConfig.dir_bert_pretrain_model, state_dict=model_state_dict,
                                         num_tag=num_tag)
    return model


def replace_useless_item(text_list, vocab):
    use_less_items = [' ', '\u3000', '\xa0', '\u2006', '\u2003', '\x85', '\u2028', '\u2002', '—', '…', '\x08', '“', '”',
                      '甍', 'Ⅱ', '\u200b', 'ヾ', '狻', '猊', 'ā', 'í', '毖', '╟', 'ლ', '筭', '詍', '瘣', '邶', '旄', '唢', '杕',
                      '烜', '╋', '≠', '弢', '㈠', '邡', '︺', '‘', 'Ç', 'à', 'È', 'ð', 'Ã', '炆', '¬', '醅', '榶', '’', '钄',
                      '\ue103', '彍', '庤', '藄', '鍏', 'ㄦ', '鑱', '鎭', '痏', '鏉', '嚜', '鍚', '屽', '煄', '泦', '臌', '珰', 'が',
                      '슈', '퍼', '맨', '이', '돌', '아', '왔', '다', '尓', '゙', '╬', '☻', 'བ', 'ོ', 'ད', 'ག', 'ཞ', 'ས', '།',
                      '魃', '鲂', '頴', '僿', '铳', '訇']
    texts = text_list.copy()
    for i, item in enumerate(texts):
        # first fan 2 jian
        if vocab.get(item) is None:
            item = zh2Hans.get(item)
            texts[i] = item
        # if can't find change to PAD
        if vocab.get(item) is None or item in use_less_items:
            texts[i] = nerConfig.PAD_flag
    return texts


def get_time():
    return time.time()


def get_time_diff(start):
    return timedelta(seconds=get_time() - start)


def conv_list_to_int(pos_list):
    res_list = []
    for pos in pos_list:
        res_list.append(int(pos))
    return res_list