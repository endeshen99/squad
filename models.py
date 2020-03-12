"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
from transformers import AlbertModel
from util import masked_softmax


class BiDAF(nn.Module):
  """Baseline BiDAF model for SQuAD.

  Based on the paper:
  "Bidirectional Attention Flow for Machine Comprehension"
  by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
  (https://arxiv.org/abs/1611.01603).

  Follows a high-level structure commonly found in SQuAD models:
      - Embedding layer: Embed word indices to get word vectors.
      - Encoder layer: Encode the embedded sequence.
      - Attention layer: Apply an attention mechanism to the encoded sequence.
      - Model encoder layer: Encode the sequence again.
      - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

  Args:
      word_vectors (torch.Tensor): Pre-trained word vectors.
      hidden_size (int): Number of features in the hidden state at each layer.
      drop_prob (float): Dropout probability.
  """

  def __init__(self, word_vectors, hidden_size, drop_prob=0.):
    super(BiDAF, self).__init__()
    self.num_labels = 2
    self.emb = AlbertModel.from_pretrained('albert-base-v1')
    self.out = nn.Linear(self.emb.config.hidden_size, self.num_labels)

    # self.emb_bert = layers.Embedding(word_vectors=self.emb.config.hidden_size,
    #                                  hidden_size = hidden_size,
    #                                  drop_prob=drop_prob)

    # self.enc = layers.RNNEncoder(input_size=hidden_size,
    #                              hidden_size=hidden_size,
    #                              num_layers=1,
    #                              drop_prob=drop_prob)

    # self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
    #                                  drop_prob=drop_prob)

    # self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
    #                              hidden_size=hidden_size,
    #                              num_layers=2,
    #                              drop_prob=drop_prob)

    # self.out = layers.BiDAFOutput(hidden_size=hidden_size,
    #                               drop_prob=drop_prob)

  def forward(self, cw_idxs, qw_idxs):
    c_mask = torch.zeros_like(cw_idxs) != cw_idxs
    q_mask = torch.zeros_like(qw_idxs) != qw_idxs

    c_type = torch.zeros_like(cw_idxs)
    q_type = torch.ones_like(qw_idxs)

    c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

    input_ids = torch.cat((cw_idxs, qw_idxs), dim=1)
    attention_mask = torch.cat((c_mask, q_mask), dim=1)
    token_type_ids = torch.cat((c_type, q_type), dim=1)

    ##print(cw_idxs.shape, qw_idxs.shape)
    emb = self.emb(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
    ##print(c_emb, q_emb)

    # c_emb_bert = self.emb_bert(c_emb)
    # q_emb_bert = self.emb_bert(q_emb)

    # c_enc = self.enc(c_emb_bert, c_len)    # (batch_size, c_len, 2 * hidden_size)
    # q_enc = self.enc(q_emb_bert, q_len)    # (batch_size, q_len, 2 * hidden_size)

    # att = self.att(c_enc, q_enc,
    #                c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

    # mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
    out = self.out(emb)  # 2 tensors, each (batch_size, c_len)

    start, end = out.split(1, dim=-1)
    start = start.squeeze(-1)
    end = end.squeeze(-1)
    # log_p1 = masked_softmax(start, torch.cat((c_mask, torch.zeros_like(qw_idxs, dtype=torch.uint8)), dim=1), log_softmax=True)
    # log_p2 = masked_softmax(end, torch.cat((c_mask, torch.zeros_like(qw_idxs, dtype=torch.uint8)), dim=1), log_softmax=True)
    return start, end
