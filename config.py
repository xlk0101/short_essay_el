import torch


class ComConfig(object):
    def __init__(self):
        self.random_seed = 515
        self.train_ratio = 0.9
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.punctuation = r"""!"#$%&'()*+,./:;<=>?@[\]^`{|}~《》。，！？‘’“”"""


class FileConfig(object):
    def __init__(self):
        # dir params
        self.dir_main = '/data/py_proj/text_proj/short_essay_el/'
        self.dir_data = self.dir_main + 'data/'
        self.dir_jieba = self.dir_data + 'jieba/'
        self.dir_kb_info = self.dir_data + 'kb_info/'
        self.dir_kb_split = self.dir_kb_info + 'split/'
        self.dir_ner = self.dir_data + 'ner/'
        self.dir_ner_split = self.dir_ner + 'split/'
        self.dir_bert = self.dir_data + 'bert/'
        self.dir_bert_pretrain_model = self.dir_bert + 'pytorch_pretrained_model/'
        self.dir_analysis = self.dir_data + 'analysis/'
        self.dir_ner_checkpoint = self.dir_data + 'ner_checkpoint/'
        self.dir_nel = self.dir_data + 'nel/'
        self.dir_tfidf = self.dir_data + 'tfidf/'
        self.dir_fasttext = self.dir_data + 'fasttext/'
        self.dir_stopword = self.dir_data + 'stopword/'
        self.dir_result = self.dir_data + 'result/'

        # file params
        self.file_kb_data = 'kb_data'
        self.file_train_data = 'train.json'
        self.file_dev_data = 'develop.json'
        self.file_stopword = 'stopword.txt'
        self.file_jieba_dict = 'jieba_dict.txt'
        self.file_kb_dict = 'kb_dict.pkl'
        self.file_kb_pandas_csv = 'pandas_kb.csv'
        self.file_kb_pandas_split = 'pandas_{}_kb.csv'
        self.file_kb_pandas_alias_data = 'pandas_kb_alias_data.csv'
        self.file_kb_pandas_alias_split = 'pandas_kb_alias_{}_split.csv'
        self.file_vocab_data = 'vocab_data.txt'
        self.file_ner_data = 'ner_data.pkl'
        self.file_ner_train_data = 'ner_train_data.pkl'
        self.file_ner_dev_data = 'ner_dev_data.pkl'
        self.file_ner_predict_data = 'ner_predict_data.pkl'
        self.file_ner_test_mention_data = 'ner_test_mention_data.pkl'
        self.file_ner_dev_mention_data = 'ner_dev_mention_data.pkl'
        self.file_ner_dev_cands_data = 'ner_dev_cands_data.json'
        self.file_ner_test_cands_data = 'ner_test_cands_data.json'
        self.file_ner_dev_mention_split = 'ner_dev_mention_split_{}.pkl'
        self.file_ner_test_mention_split = 'ner_test_mention_split_{}.pkl'
        self.file_ner_dev_cands_split = 'ner_dev_cands_split_{}.pkl'
        self.file_ner_test_cands_split = 'ner_test_cands_split_{}.pkl'
        self.file_bert_vocab = 'vocab.txt'
        self.file_analysis_vocab = 'analysis_vocab.txt'
        self.file_analysis_unfind = 'analysis_unfind.txt'
        self.file_ner_model = 'ner_model.bin'
        self.file_ner_predict_tag = 'ner_predict_tag.pkl'
        self.file_ner_test_predict_tag = 'ner_test_predict_tag.pkl'
        self.file_nel_train_data = 'nel_train_data.pkl'
        self.file_nel_entity_link_train_data = 'nel_entity_link_train_data.pkl'
        self.file_nel_mention_context_vocab = 'nel_mention_context_vocab.txt'
        self.file_nel_entity_context_vocab = 'nel_entity_context_vocab.txt'
        self.file_nel_entity_vocab = 'nel_entity_vocab.txt'
        self.file_tfidf_save_data = 'tfidf_save_data.pkl'
        self.file_fasttext_train_data = 'fasttext_train_data.txt'
        self.file_fasttext_model = 'fasttext_{}_model.bin'
        self.file_result_fasttext_predict = 'result_fasttext_predict.txt'
        self.file_result_fasttext_test = 'result_fasttext_test.txt'


class NERConfig(object):
    def __init__(self):
        self.B_seg = 'B_'
        self.I_seg = 'I_'
        self.E_seg = 'E_'
        self.O_seg = 'O'
        self.KB_seg = 'KB'
        self.NIL_seg = 'NIL'
        self.PAD_flag = '[PAD]'
        self.CLS_flag = '[CLS]'
        self.SEP_flag = '[SEP]'
        self.UNK_flag = '[UNK]'
        self.mode_train = 'train'
        self.mode_dev = 'dev'
        self.mode_predict = 'predict'
        self.mode_test = 'test'
        self.ner_task_name = 'ner'
        # self.labels = ["B_KB", "I_KB", "E_KB", "B_NIL", "I_NIL", "E_NIL", "O"]
        self.labels = ["B_KB", "I_KB", "E_KB", "O"]
        # NER model
        self.max_seq_length = 51
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.learning_rate = 5e-5
        self.num_train_epochs = 3
        self.warmup_proportion = 0.1
        self.gradient_accumulation_steps = 1
        self.fp16 = False
        self.loss_scale = 0.


class NELConfig(object):
    def __init__(self):
        self.max_cands_num = 10
        self.mode_train = 'train'
        self.mode_predict = 'predict'
        self.batch_size = 16
        self.emb_dim = 128
        self.hid_dim = 256
        self.train_epoches = 10
        self.n_layers = 1


class TfIdfConfig(object):
    def __init__(self):
        self.start_index = 10001
        self.is_use_local_dict = True


class FastTextConfig(object):
    def __init__(self):
        self.model_skipgram = 'skipgram'
        self.model_cbow = 'cbow'
        self.max_alias_num = 20
        self.min_entity_similarity_threshold = 0.5
