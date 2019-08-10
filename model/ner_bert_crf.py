import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.metrics import f1_score, classification_report
from model.ner_crf import CRF
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class Ner_Bert_Crf(BertPreTrainedModel):
    def __init__(self, model_config, num_tag):
        super(Ner_Bert_Crf, self).__init__(model_config)
        self.bert = BertModel(model_config)
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        # bilstm
        # self.bilstm = nn.LSTM(model_config.hidden_size, model_config.hidden_size, batch_first=True,
        #                       bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(model_config.hidden_size, num_tag)
        self.apply(self.init_bert_weights)
        self.crf = CRF(num_tag)

    def forward(self, input_ids, token_type_ids, attention_mask, label_id=None, output_all_encoded_layers=False,
                input_length=None):
        bert_encode, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                   output_all_encoded_layers=output_all_encoded_layers)
        # bilstm
        # packed = pack_padded_sequence(bert_encode, input_length, batch_first=True, enforce_sorted=False)
        # rnn_out, _ = self.bilstm(packed)
        # # rnn_out:[B, L, hidden_size*2]
        # rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        # bilstm
        out = self.dropout(bert_encode)  # add dropout
        output = self.classifier(out)
        # output = self.classifier(rnn_out)
        return output

    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask, is_squeeze=True):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        if is_squeeze:
            predicts = predicts.view(1, -1).squeeze()
            predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
