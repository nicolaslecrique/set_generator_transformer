import math

import torch
from torch import nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# This is loosely inspired by the code available here:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):

    def __init__(self,
                 idx_to_token: [str],
                 embedding_dim_between_layers: int,
                 nb_attention_heads: int,
                 hidden_dim_feed_forward_layers: int,
                 nb_encoder_layers: int,
                 padding_idx: int,
                 add_positioning_to_embeddings: bool = True,
                 dropout: float = 0.5,
                 max_sentence_size: int=50):

        super(TransformerModel, self).__init__()
        self.src_mask_by_sequence_size = [self._generate_square_subsequent_mask(len_sequence) for len_sequence in range(max_sentence_size + 1)]
        self.vocab_to_embedding = nn.Embedding(len(idx_to_token), embedding_dim_between_layers, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim_between_layers) if add_positioning_to_embeddings else None

        encoder_layers = TransformerEncoderLayer(
            embedding_dim_between_layers,
            nb_attention_heads,
            hidden_dim_feed_forward_layers,
            dropout
        )

        self.transformer_encoder = TransformerEncoder(encoder_layers, nb_encoder_layers)
        self.embedding_to_vocab = nn.Linear(embedding_dim_between_layers, len(idx_to_token))
        self.embedding_dim_between_layers = embedding_dim_between_layers
        self.idx_to_token = idx_to_token  # so it's serialized with model
        self._init_weights()

    # Generate a mask for the TransformerEncoderLayer
    # it allows the generator to use previous but not following tokens to generate the current one.
    def _generate_square_subsequent_mask(self, len_sequence):
        mask = (torch.triu(torch.ones(len_sequence, len_sequence)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_weights(self):
        initrange = 0.1
        self.vocab_to_embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_to_vocab.bias.data.zero_()
        self.embedding_to_vocab.weight.data.uniform_(-initrange, initrange)

    # take a tensor of dimensions [sequence_size, batch_size]
    # sequences is a [sequence_size, batch_size] tensor. Each entry is a token index
    def forward(self, sequences):

        # 1. We transform each token to its corresponding embedding.
        # We get a tensor of dimensions [sequence_size, batch_size, embedding_size]
        # See http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder for the normalization factor
        sequences = self.vocab_to_embedding(sequences) * math.sqrt(self.embedding_dim_between_layers)

        # 2. We add positional encoding to the embeddings
        if self.pos_encoder:
            sequences = self.pos_encoder(sequences)

        # 3. We pass the sequence through the encoder layers
        # We get a tensor of dimension [sequence_size, batch_size, embedding_dim_between_layers]
        output = self.transformer_encoder(sequences, self.src_mask_by_sequence_size[len(sequences)])

        # 4. We convert output embeddings to vocabulary size vectors that can be processed by the softmax layer
        # We get a tensor of dimension [sequence_size, batch_size, vocabulary_size]
        # the softmax layer is inside the loss function (as it does not contain learnable parameters)
        vocab = self.embedding_to_vocab(output)
        return vocab


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


