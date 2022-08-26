from torch import nn as nn

from generator_model.transformer.transformer_model import TransformerModel


# output of transformer is passed to a linear layer embedding_size -> vocab_size for next token prediction task
# similar to any language model, like GPT-2
class LanguageModelTransformer(nn.Module):

    def __init__(self, transformer: TransformerModel, idx_to_token: [str]):
        super(LanguageModelTransformer, self).__init__()
        self.transformer = transformer
        self.embedding_to_vocab = nn.Linear(transformer.output_dim, transformer.nb_tokens_in_vocab)
        self.idx_to_token = idx_to_token  # so it's serialized with model

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding_to_vocab.bias.data.zero_()
        self.embedding_to_vocab.weight.data.uniform_(-initrange, initrange)

    def forward(self, sequences):
        output_embedding = self.transformer(sequences)
        vocab = self.embedding_to_vocab(output_embedding)
        return vocab
