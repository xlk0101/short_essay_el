import sys
import config

from evaluate import ner_evaluate
from utils import data_utils, com_utils
from utils.com_utils import ProgressBar
from model.ner_bert_crf import Ner_Bert_Crf

# init params
fileConfig = config.FileConfig()
nerConfig = config.NERConfig()


def train():
    train_iter, num_train_steps = data_utils.create_ner_batch_iter(nerConfig.mode_extend_train)
    eval_iter = data_utils.create_ner_batch_iter(nerConfig.mode_extend_dev)
    epoch_size = num_train_steps * nerConfig.train_batch_size * nerConfig.gradient_accumulation_steps / nerConfig.num_train_epochs
    pbar = ProgressBar(epoch_size=epoch_size, batch_size=nerConfig.train_batch_size)
    model = Ner_Bert_Crf.from_pretrained(fileConfig.dir_bert_pretrain_model, num_tag=len(nerConfig.labels))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    ner_evaluate.fit_eval(model=model, training_iter=train_iter, eval_iter=eval_iter,
                          num_epoch=nerConfig.num_train_epochs, pbar=pbar,
                          num_train_steps=num_train_steps, verbose=1)


def predict():
    dev_iter, data_list = data_utils.create_ner_batch_iter(nerConfig.mode_predict)
    model = com_utils.load_model(fileConfig.dir_ner_checkpoint, num_tag=len(nerConfig.labels))
    predict_list = ner_evaluate.predict(model, dev_iter)
    data_utils.deal_ner_predict_data(predict_list, data_list,fileConfig.dir_ner + fileConfig.file_ner_predict_tag)
    print("success predict dev data")


def test():
    test_iter, data_list = data_utils.create_ner_batch_iter(nerConfig.mode_test)
    model = com_utils.load_model(fileConfig.dir_ner_checkpoint, num_tag=len(nerConfig.labels))
    predict_list = ner_evaluate.predict(model, test_iter)
    data_utils.deal_ner_predict_data(predict_list, data_list, fileConfig.dir_ner + fileConfig.file_ner_test_predict_tag)
    print("success predict test data")


if __name__ == '__main__':
    if len(sys.argv) == 1 or not sys.argv[1] in ['train', 'test', 'predict']:
        print("should input param [train/test/predict]")
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'predict':
        predict()
    elif sys.argv[1] == 'test':
        test()
    else:
        pass
