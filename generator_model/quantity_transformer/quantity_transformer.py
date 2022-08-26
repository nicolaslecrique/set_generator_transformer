import torch
from torch import nn as nn

from generator_model.transformer.transformer_model import TransformerModel


# As in Bert paper, we use the first token output for a regression and the following, https://arxiv.org/abs/1810.04805
# the others papers are a standard linear output of transformer, without positioning to make it a set transformer, in the spirit of https://arxiv.org/abs/1810.00825
# this is case of multi-task learning: https://arxiv.org/pdf/1707.08114.pdf
class QuantityTransformerBatchResult:

    # tensor (sequence_size - 1, batch_size); -1 because first token is sos and doesn't belong to softmax
    linear_output_for_softmax: torch.Tensor
    # tensor (batch_size)
    regression_output: torch.Tensor

    def __init__(
            self,
            linear_output_for_softmax: torch.Tensor,
            regression_output: torch.Tensor
    ):
        self.linear_output_for_softmax = linear_output_for_softmax
        self.regression_output = regression_output


class QuantityTransformer(nn.Module):

    def __init__(
            self,
            transformer: TransformerModel,
            padding_token_idx: int,
            idx_to_token: [str],
            weight_normalization_factor: float
    ):
        super(QuantityTransformer, self).__init__()
        self.transformer = transformer
        self.embedding_to_scalar_for_softmax = nn.Linear(transformer.output_dim, 1)
        self.embedding_to_scalar_for_regression = nn.Linear(transformer.output_dim, 1)
        self.idx_to_token = idx_to_token  # so it's serialized with model
        self.weight_normalization_factor = weight_normalization_factor

        self.padding_token_idx = padding_token_idx

    # sequences: (seq_size, batch_size) or one-hot
    def forward(self, sequences: torch.Tensor) -> QuantityTransformerBatchResult:
        output_embedding: torch.Tensor = self.transformer(sequences)

        embedding_for_regression = output_embedding[0]
        embedding_for_softmax = output_embedding[1:]

        # (sequence, batch, 1)
        softmax_scalar_by_sequence_idx_and_batch = self.embedding_to_scalar_for_softmax(embedding_for_softmax)

        # (batch, 1)
        regression_scalar_by_batch = self.embedding_to_scalar_for_regression(embedding_for_regression)

        # remove embedding dimension (sequence, batch, 1) => (sequence, batch)
        softmax_squeezed = softmax_scalar_by_sequence_idx_and_batch.squeeze(2)

        # (batch)
        regression_squeezed = regression_scalar_by_batch.squeeze()

        return QuantityTransformerBatchResult(
            linear_output_for_softmax=softmax_squeezed,
            regression_output=regression_squeezed)
