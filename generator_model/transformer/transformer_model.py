import math

import torch
from torch import nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# transfomer architecture, inspired by code example in https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# cf. https://arxiv.org/abs/1706.03762 (attention is all you need) and
# https://arxiv.org/pdf/1810.04805.pdf (BERT model)
class TransformerModel(nn.Module):

    nb_tokens_in_vocab: int
    output_dim: int

    def __init__(self,
                 nb_tokens_in_vocab: int,
                 embedding_dim_between_layers: int,
                 nb_attention_heads: int,
                 hidden_dim_feed_forward_layers: int,
                 nb_encoder_layers: int,
                 padding_idx: int,
                 mask_next_tokens: bool,
                 positioning: bool=True,
                 dropout: float=0.5,
                 max_sentence_size: int=50):

        super(TransformerModel, self).__init__()
        self.positioning = positioning
        self.model_type = 'Transformer'
        self.output_dim = embedding_dim_between_layers
        self.nb_tokens_in_vocab = nb_tokens_in_vocab
        self.mask_next_tokens = mask_next_tokens
        if mask_next_tokens:
            self.src_mask_by_sequence_size = [None] * (max_sentence_size + 1)

        self.pos_encoder_or_dropout = PositionalEncoding(embedding_dim_between_layers, dropout) if positioning else nn.Dropout(p=dropout)

        encoder_layers = TransformerEncoderLayer(
            embedding_dim_between_layers,
            nb_attention_heads,
            hidden_dim_feed_forward_layers,
            dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nb_encoder_layers)
        self.vocab_to_embedding = nn.Embedding(nb_tokens_in_vocab, embedding_dim_between_layers, padding_idx=padding_idx)
        self.embedding_dim_between_layers = embedding_dim_between_layers

        self._init_weights()

    def _generate_square_subsequent_mask(self, len_sequence):
        mask = (torch.triu(torch.ones(len_sequence, len_sequence)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_weights(self):
        initrange = 0.1
        self.vocab_to_embedding.weight.data.uniform_(-initrange, initrange)

    # take a tensor of size [sequence_size, batch_size]
    def forward(self, sequences):
        len_sequence = len(sequences)

        mask = None
        if self.mask_next_tokens:
            if self.src_mask_by_sequence_size[len_sequence] is None:
                device = sequences.device
                mask = self._generate_square_subsequent_mask(len_sequence).to(device)
                self.src_mask_by_sequence_size[len_sequence] = mask
            mask = self.src_mask_by_sequence_size[len_sequence]

        # division is in tutorial and also done in http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder
        # but it's explained nowhere !!! is it for the same reason that it's in attention ?
        # is it equivalent to batch normalization fo embedding ?

        sequences = self.vocab_to_embedding(sequences) * math.sqrt(self.embedding_dim_between_layers)
        sequences = self.pos_encoder_or_dropout(sequences)
        output = self.transformer_encoder(sequences, mask)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


