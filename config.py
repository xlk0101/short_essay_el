import torch


class ComConfig(object):
    def __init__(self):
        self.random_seed = 515
        self.train_ratio = 0.9
        self.max_extend_mentions_num = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.punctuation = r"""!"#$%&'()*+,./:;<=>?@[\]^`{|}~《》。·，！？‘’“”"""
        self.re_find_brackets = r'[{}](.*?)[{}]'
        self.brackets_list = ['<>', '()', '《》', '\'\'', '\"\"', '{}', '[]']
        self.mode_ner_normal = 'normal'
        self.mode_ner_extend = 'extend'
        self.create_ner_mode = self.mode_ner_extend


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
        self.dir_tmp = self.dir_data + 'tmp/'

        # file params
        self.file_kb_data = 'kb_data'
        self.file_train_data = 'train.json'
        self.file_extend_train_data = 'extend_train.json'
        self.file_train_data_split = 'train_data_{}_split.txt'
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
        self.file_extend_ner_data = 'extend_ner_data.pkl'
        self.file_ner_train_data = 'ner_train_data.pkl'
        self.file_ner_extend_train_data = 'ner_extend_train_data.pkl'
        self.file_ner_dev_data = 'ner_dev_data.pkl'
        self.file_ner_extend_dev_data = 'ner_extend_dev_data.pkl'
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
        self.file_analysis_train_untind = 'analysis_train_unfind.txt'
        self.file_ner_model = 'ner_model.bin'
        self.file_ner_predict_tag = 'ner_predict_tag.pkl'
        self.file_ner_test_predict_tag = 'ner_test_predict_tag.pkl'
        self.file_nel_train_data = 'nel_train_data.pkl'
        self.file_nel_entity_link_train_data = 'nel_entity_link_train_data.pkl'
        self.file_nel_mention_context_vocab = 'nel_mention_context_vocab.txt'
        self.file_nel_entity_context_vocab = 'nel_entity_context_vocab.txt'
        self.file_nel_entity_vocab = 'nel_entity_vocab.txt'
        self.file_tfidf_save_data = 'tfidf_save_data.pkl'
        self.file_fasttext_unsup_train_data = 'fasttext_unsup_train_data.txt'
        self.file_fasttext_sup_train_data = 'fasttext_sup_train_data.txt'
        self.file_fasttext_sup_test_data = 'fasttext_sup_test_data.txt'
        self.file_fasttext_sup_train_split = 'fasttext_sup_train_{}_split.txt'
        self.file_fasttext_model = 'fasttext_{}_model.bin'
        self.file_fasttext_sup_model = 'fasttext_sup_model.bin'
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
        self.mode_extend_train = 'extend_train'
        self.mode_dev = 'dev'
        self.mode_extend_dev = 'extend_dev'
        self.mode_predict = 'predict'
        self.mode_test = 'test'
        self.ner_task_name = 'ner'
        # self.labels = ["B_KB", "I_KB", "E_KB", "B_NIL", "I_NIL", "E_NIL", "O"]
        self.labels = ["B_KB", "I_KB", "E_KB", "O"]
        # self.labels = ["E_Human", "E_Event", "E_ScientificOrganization", "E_Movie", "E_Thing", "I_CreativeWork",
        #                "E_FictionalHuman", "B_CommunicationMedium", "I_CommunicationMedium", "E_CreativeWork",
        #                "I_Thing", "E_EntertainmentPerson", "B_Movie", "E_Product", "I_Movie", "I_Event", "O",
        #                "B_Vocabulary", "E_CommunicationMedium", "B_Event", "B_Human", "B_Organization", "I_Place",
        #                "E_Organization", "B_Thing", "E_Place", "I_Vocabulary", "I_FictionalHuman",
        #                "I_ScientificOrganization", "E_Vocabulary", "B_Product", "B_EntertainmentPerson",
        #                "B_ScientificOrganization", "I_Product", "I_Human", "I_EntertainmentPerson", "I_Organization",
        #                "B_CreativeWork", "B_FictionalHuman", "B_Place"]
        self.ner_type_map = {
            'FamilyName-HistoricalPeriod-Vocabulary-Dynasty': 'Vocabulary',
            'Organization-Product-Brand': 'Organization',
            'Currency-FictionalThing-Vocabulary': 'Vocabulary',
            'Building-Place-RealEstate': 'Place',
            'Tool-Product-Vocabulary': 'Thing',
            'Organization-Brand-Product': 'Organization',
            'Product-Material-Vocabulary': 'Product',
            'Organization-CreativeWork-Place': 'Organization',
            'CreativeWork-FictionalThing-Organism': 'CreativeWork',
            'Product-Tool-Vocabulary': 'Thing',
            'HistoricalPeriod-Vocabulary-Dynasty': 'Thing',
            'Tool-CreativeWork-Product': 'Thing',
            'Organization-Brand-Place': 'Organization',
            'Building-Place-Vocabulary': 'Place',
            'Product-Place-Vocabulary': 'Product',
            'CommunicationMedium-CreativeWork-Event': 'CommunicationMedium',
            'Organization-Vocabulary-HistoricalPeriod': 'Organization',
            'Food-Organization-Brand': 'Thing',
            'Material-Vocabulary-ChemicalElement': 'Vocabulary',
            'Organization-Place-Vocabulary': 'Organization',
            'CulturalHeritage-CreativeWork-Vocabulary': 'CreativeWork',
            'Food-Material-Vocabulary': 'Thing',
            'Game-FictionalThing-Tool': 'CreativeWork',
            'Currency-Product-Vocabulary': 'Product',
            'Tool-FamilyName-Vocabulary': 'Thing',
            'Food-Organism-Vocabulary': 'Thing',
            'Dynasty-Place-HistoricalPeriod': 'Thing',
            'Product-CreativeWork-Vocabulary': 'Product',
            'Building-Organization-Place': 'Place',
            'AcademicDiscipline-Curriculum-EducationMajor': 'CreativeWork',
            'AcademicDiscipline-Vocabulary-EducationMajor': 'CreativeWork',
            'HistoricalPeriod-Place-Dynasty': 'Thing',
            'Currency-Product-Organization': 'Product',
            'Building-CulturalHeritage-Place': 'Place',
            'Organization-CommunicationMedium-CreativeWork': 'Organization',
            'Currency-Product-CreativeWork': 'Product',
            'Organism-FictionalThing-CreativeWork': 'CreativeWork',
            'Organization-Brand-Vocabulary': 'Organization',
            'CulturalHeritage-Tool-CreativeWork': 'Thing',
            'CulturalHeritage-CommunicationMedium-CreativeWork': 'CommunicationMedium',
            'HistoricalPerson-Human': 'Human',
            'Athlete-Human': 'Human',
            'CommunicationMedium-CreativeWork': 'CommunicationMedium',
            'Human-Athlete': 'Human',
            'Human-HistoricalPerson': 'Human',
            'ScientificOrganization-CollegeOrUniversity': 'ScientificOrganization',
            'CollegeOrUniversity-ScientificOrganization': 'ScientificOrganization',
            'CreativeWork-Vocabulary': 'CreativeWork',
            'Organization-Place': 'Organization',
            'Building-Place': 'Place',
            'CreativeWork-Event': 'CreativeWork',
            'CommunicationMedium-Curriculum': 'CommunicationMedium',
            'Event-Vocabulary': 'Vocabulary',
            'Product-FictionalThing': 'CreativeWork',
            'Organization-Brand': 'Organization',
            'FamilyName-Vocabulary': 'Vocabulary',
            'Place-RealEstate': 'Place',
            'Organization-CommunicationMedium': 'Organization',
            'Brand-Product': 'Product',
            'HistoricalPeriod-Dynasty': 'Thing',
            'Place-Vocabulary': 'Place',
            'Person-Organization': 'Human',
            'Organization-Vocabulary': 'Organization',
            'CulturalHeritage-CreativeWork': 'CreativeWork',
            'Product-Vocabulary': 'Product',
            'EntertainmentPerson-Human': 'Human',
            'CommunicationMedium-Vocabulary': 'CommunicationMedium',
            'CommunicationMedium-Brand': 'CommunicationMedium',
            'Language-Vocabulary': 'Vocabulary',
            'Organization-Event': 'Organization',
            'Human-EntertainmentPerson': 'Human',
            'Material-ChemicalElement': 'Product',
            'Building-Organization': 'Organization',
            'Game-Product': 'CreativeWork',
            'Game-FictionalThing': 'CreativeWork',
            'CommunicationMedium-Event': 'CommunicationMedium',
            'MedicalCondition-Vocabulary': 'Vocabulary',
            'Product-Game': 'CreativeWork',
            'Tool-Vocabulary': 'Thing',
            'Product-Tool': 'Thing',
            'AcademicDiscipline-Vocabulary': 'CreativeWork',
            'AcademicDiscipline-EducationMajor': 'CreativeWork',
            'Organism-Vocabulary': 'Vocabulary',
            'Organization-Game': 'CreativeWork',
            'Product-CreativeWork': 'CreativeWork',
            'Food-Vocabulary': 'Thing',
            'MedicalCondition-Event': 'Event',
            'Organization-Person': 'Human',
            'CulturalHeritage-Event': 'Event',
            'FictionalThing-CreativeWork': 'CreativeWork',
            'Currency-FictionalThing': 'CreativeWork',
            'Organization-CreativeWork': 'CreativeWork',
            'Person-Product': 'Product',
            'AcademicDiscipline-Curriculum': 'CreativeWork',
            'Food-Brand': 'Thing',
            'Food-Material': 'Thing',
            'Vocabulary-Theorem': 'Vocabulary',
            'Product-CommunicationMedium': 'CommunicationMedium',
            'Material-Vocabulary': 'Vocabulary',
            'Material-CreativeWork': 'CreativeWork',
            'FictionalThing-Vocabulary': 'CreativeWork',
            'Product-Brand': 'Product',
            'Currency-Product': 'Product',
            'Brand-Material': 'Vocabulary',
            'Food-Organism': 'Thing',
            'CreativeWork-Organism': 'CreativeWork',
            'Organization-Organism': 'Organization',
            'Vocabulary-HistoricalPeriod': 'Thing',
            'Tool-CreativeWork': 'Thing',
            'AstronomicalObject-CreativeWork': 'CreativeWork',
            'CommunicationMedium-Place': 'CommunicationMedium',
            'Product-Material': 'Product',
            'AcademicDiscipline-CommunicationMedium': 'CreativeWork',
            'AstronomicalObject-Vocabulary': 'Vocabulary',
            'Tool-Product': 'Thing',
            'Tool-FictionalThing': 'Thing',
            'Brand-Vocabulary': 'Vocabulary',
            'Game-Event': 'CreativeWork',
            'Organization-Product': 'Organization',
            'Game-Vocabulary': 'CreativeWork',
            'Place-CreativeWork': 'CreativeWork',
            'Game-CreativeWork': 'CreativeWork',
            'Curriculum-Vocabulary': 'Vocabulary',
            'Vocabulary-Language': 'Vocabulary',
            'FictionalThing-Place': 'Place',
            'Person-Vocabulary': 'Human',
            'Product-Event': 'Product',
            'Tool-Event': 'Thing',
            'CulturalHeritage-Vocabulary': 'Vocabulary',
            'CreativeWork-Place': 'CreativeWork',
            'Game-HistoricalPeriod': 'CreativeWork',
            'CulturalHeritage-Place': 'Place',
            'MedicalCondition-Organism': 'Thing',
            'Vocabulary-EducationMajor': 'Vocabulary',
            'Organization-FictionalThing': 'Organization',
            'Person-CreativeWork': 'Human',
            'FictionalThing-Organism': 'Thing',
            'Brand-CreativeWork': 'CreativeWork',
            'Currency-Vocabulary': 'Vocabulary',
            'Organization-Tool': 'Thing',
            'CommunicationMedium-EducationMajor': 'CommunicationMedium',
            'Currency-CreativeWork': 'CreativeWork',
            'CreativeWork-HistoricalPeriod': 'Thing',
            'Person-CommunicationMedium': 'Human',
            'Thing': 'Thing',
            'CreativeWork': 'CreativeWork',
            'Plant': 'Thing',
            'Human': 'Human',
            'Place': 'Place',
            'Movie': 'Movie',
            'EntertainmentPerson': 'EntertainmentPerson',
            'FictionalHuman': 'FictionalHuman',
            'CommunicationMedium': 'CommunicationMedium',
            'Vocabulary': 'Vocabulary',
            'Game': 'CreativeWork',
            'TVShow': 'CreativeWork',
            'TVPlay': 'CreativeWork',
            'Tool': 'Thing',
            'Material': 'Thing',
            'Event': 'Event',
            'Organization': 'Organization',
            'Language': 'Vocabulary',
            'Product': 'Product',
            'Organism': 'Thing',
            'Animal': 'Thing',
            'Brand': 'Thing',
            'ScientificOrganization': 'ScientificOrganization',
            'Country': 'Place',
            'MedicalCondition': 'Event',
            'Person': 'Human',
            'Food': 'Thing',
            'FictionalThing': 'CreativeWork',
            'CulturalHeritage': 'CreativeWork',
            'FamilyName': 'Vocabulary',
            'ZodiacSign': 'CreativeWork',
            'Nation': 'Place',
            'AstronomicalObject': 'Thing',
            'Symbol': 'CreativeWork',
            'Currency': 'Product',
            'AcademicDiscipline': 'CreativeWork',
            'CollegeOrUniversity': 'ScientificOrganization',
            'Curriculum': 'CreativeWork',
            'EducationMajor': 'Thing',
            'AwardEventSeries': 'CreativeWork',
            'InternationalOrganization': 'Organization',
            'HistoricalPeriod': 'Thing',
            'Formula': 'CreativeWork',
            'MedicalDepartmentType': 'Organization',
            'Theorem': 'Vocabulary',
            'Athlete': 'Human',
            'HistoricalPerson': 'Human'
        }
        # NER model
        self.max_seq_length = 51
        self.train_batch_size = 48
        self.eval_batch_size = 48
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
        self.label_true = '__label__true'
        self.label_false = '__label__false'
        self.max_alias_num = 30
        self.min_entity_similarity_threshold = 0.4
        self.choose_entity_similarity_threshold = 0.7
