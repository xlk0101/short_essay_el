import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from utils import data_utils
import config

# init params
comConfig = config.ComConfig()
nelConfig = config.NELConfig()


class EntityLink(nn.Module):
    def __init__(self, mention_token_vocab_size, entity_token_vocab_size, entity_vocab_size, emb_dim, hid_dim):
        super(EntityLink, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=nelConfig.n_layers, bidirectional=True, dropout=0.0,
                            batch_first=True)
        self._init_h = nn.Parameter(
            torch.Tensor(nelConfig.n_layers * 2, hid_dim).to(comConfig.device))  # 默认是bidirectional
        self._init_c = nn.Parameter(torch.Tensor(nelConfig.n_layers * 2, hid_dim).to(comConfig.device))
        self.mention_vocab_size = mention_token_vocab_size
        self.entity_vocab_size = entity_token_vocab_size
        self.embedding = nn.Embedding(entity_vocab_size, emb_dim)
        self.mention_embedding = nn.Embedding(self.mention_vocab_size, emb_dim)
        self.entity_embedding = nn.Embedding(self.entity_vocab_size, emb_dim)

        self.attention_w = nn.Linear(2 * hid_dim, 1)
        self.reduce_w = nn.Linear(2 * hid_dim, emb_dim)
        self.reduce_m = nn.Linear(6 * hid_dim, emb_dim)
        self.w_e = nn.Linear(emb_dim, emb_dim)
        self.w_c = nn.Linear(emb_dim, emb_dim)
        self.b = nn.Parameter(torch.Tensor([emb_dim]))
        self.v = nn.Linear(emb_dim, 1)

        # init vector
        init.orthogonal_(self._init_h)
        init.orthogonal_(self._init_c)
        init.normal_(self.mention_embedding.weight, -0.01, 0.01)
        init.normal_(self.entity_embedding.weight, -0.01, 0.01)
        init.xavier_normal_(self.attention_w.weight)
        init.xavier_normal_(self.reduce_w.weight)
        init.xavier_normal_(self.reduce_m.weight)
        init.xavier_normal_(self.w_e.weight)
        init.xavier_normal_(self.w_c.weight)
        init.normal_(self.b)
        init.xavier_normal_(self.v.weight)

    def soft_attention(self, embs, poss):
        res = []
        # print(f'emb size : {embs.size()} poss size : {poss.size()} ')
        batch_size = embs.size(0)
        for (emb, pos) in zip(embs.chunk(chunks=batch_size, dim=0), poss.chunk(chunks=batch_size, dim=0)):
            emb = emb.squeeze(0)
            pos = pos.squeeze(0)
            emb = emb[pos[0]:pos[1] + 1, :]
            attention = self.attention_w(emb)
            attention = F.softmax(attention, dim=0)
            res.append(torch.sum(attention * emb, dim=0))
        return torch.stack(res, dim=0)  # batch, hid_dim

    def generate_representation(self, context_emb, position):
        '''
        :param context_emb: batch, seq_len, emb_dim
        :param position: batch, 2
        :return:
        '''
        assert context_emb.size(0) == position.size(0)
        position_expand = position.unsqueeze(-1).repeat(1, 1, context_emb.size(-1))
        emb = torch.gather(context_emb, 1, position_expand)  # batch, 2, hid_dim
        soft_attention_emb = self.soft_attention(context_emb, position)
        soft_attention_emb = soft_attention_emb.unsqueeze(1)
        res = torch.cat((emb, soft_attention_emb), dim=1)  # batch, 3, hid_dim
        res = res.view(context_emb.size(0), -1)  # batch, 3*hid_dim
        return res.contiguous()

    def lstm_forward(self, emb_input, input_lens):
        # emb_input [batch, seq_len, emb_dim]
        batch_size = emb_input.size(0)
        encoder_output, encoder_states = lstm_encoder(emb_input, self.lstm, input_lens,
                                                      (self._init_h.unsqueeze(1).repeat(1, batch_size, 1),
                                                       self._init_c.unsqueeze(1).repeat(1, batch_size, 1)),
                                                      is_mask=True)
        return encoder_output, encoder_states

    def forward(self, mention_contexts, mention_position, entity_contexts, entities, target=None):
        """
         mention_contexts : batch, m_seq_len
         mention_position: batch, 2 (start, end)
         entity_contexts : batch, candidates, e_seq_len
         entities : batch, candidates
         target : batch
        """
        mention_seq_len = torch.sum(mention_contexts > 0, dim=-1)
        # mention_seq_len = mention_seq_len[mention_seq_len > 0]
        mention_contexts_emb = self.mention_embedding(mention_contexts)
        # filter 0 rows
        # select_tensor = mention_seq_len > 0
        # mention_seq_len = mention_seq_len[select_tensor]
        # mention_contexts = mention_contexts[select_tensor]
        mention_contexts_emb, _ = self.lstm_forward(mention_contexts_emb, mention_seq_len)
        mention_emb = self.generate_representation(mention_contexts_emb, mention_position)
        entity_embs = []
        cands_num = entity_contexts.size(1)
        for entity_context, entity in zip(entity_contexts.chunk(dim=1, chunks=cands_num),
                                          entities.chunk(dim=1, chunks=cands_num)):
            entity_context = entity_context.squeeze(1)
            entity = entity.squeeze(1)
            entity_seq_len = torch.sum(entity_context > 0, dim=1)
            # filter 0 rows
            # select_tensor = entity_seq_len > 0
            # entity_seq_len = entity_seq_len[select_tensor]
            # entity_context = entity_context[select_tensor]
            # entity = entity[select_tensor]
            entity_contexts_emb = self.entity_embedding(entity_context)  # batch, e_seq_len, emb_dim
            entity_emb = self.embedding(entity)  # batch, emb_dim
            entity_contexts_emb, _ = self.lstm_forward(entity_contexts_emb, entity_seq_len)
            entity_contexts_emb = F.relu(self.reduce_w(entity_contexts_emb))  # batch, e_seq_len, emb_dim
            # add attention
            entity_emb = entity_emb.unsqueeze(1).expand_as(entity_contexts_emb)
            # torch.sum( * entity_contexts_emb, dim=-1)
            attention = self.v(
                torch.tanh(self.w_c(entity_contexts_emb) + self.w_e(entity_emb) + self.b))  # batch, e_seq_len, 1
            attention = F.softmax(attention, dim=1)
            entity_contexts_emb = torch.sum(attention * entity_contexts_emb, dim=1)  # batch, emb_dim
            entity_embs.append(entity_contexts_emb)
        entity_embs = torch.stack(entity_embs, dim=1)  # batch, cands, emb_dim
        mention_emb = F.relu(self.reduce_m(mention_emb))  # batch, emb_dim
        scores = torch.sum(mention_emb.unsqueeze(1) * entity_embs, dim=-1)  # batch, cands
        scores = F.softmax(scores, dim=-1)
        if target is not None:
            loss = F.multi_margin_loss(scores, target, margin=0.5)
            return scores, loss
        else:
            return scores


def lstm_encoder(sequence, lstm, seq_lens, init_states, is_mask=False, get_final_output=False):
    batch_size = sequence.size(0)
    if isinstance(seq_lens, torch.Tensor):
        seq_lens_value = seq_lens.tolist()
    else:
        seq_lens_value = seq_lens
    assert len(seq_lens_value) == batch_size
    sort_ind = np.argsort(seq_lens_value)[::-1].tolist()
    sort_seq_lens = [seq_lens_value[i] for i in sort_ind]
    emb_sequence = data_utils.reorder_sequence(sequence, sort_ind)
    init_states = (init_states[0].contiguous(), init_states[1].contiguous())
    packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence, sort_seq_lens, batch_first=True)
    packed_out, final_states = lstm(packed_seq, init_states)
    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
    back_map = {ind: i for i, ind in enumerate(sort_ind)}
    reorder_ind = [back_map[i] for i in range(len(sort_ind))]
    lstm_out = data_utils.reorder_sequence(lstm_out, reorder_ind)
    final_states = data_utils.reorder_lstm_states(final_states, reorder_ind)
    if is_mask:
        mask = data_utils.get_mask(seq_lens)  # batch, max_seq_lens
        assert lstm_out.size(1) == mask.size(1)
        lstm_out *= mask.unsqueeze(-1)
        return lstm_out, final_states  # [batch, max_seq_lens, hid_dim], ([n_layer, batch, hid_dim], [n_layer, batch, hid_dim])
    if get_final_output:
        mask = data_utils.get_mask(seq_lens)
        lstm_out = data_utils.get_final_encoder_states(lstm_out, mask, bidirectional=True)
        return lstm_out, final_states  # [batch, hid_dim], ([n_layer, batch, hid_dim], [n_layer, batch, hid_dim])
