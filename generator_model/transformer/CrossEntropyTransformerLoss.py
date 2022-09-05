import torch
from torch import nn as nn
from torch.nn import functional as F

from generator_model.transformer.transformer_loss import TransformerLoss, TransformerLossResult


class CrossEntropyLossTransformerLoss(TransformerLoss):

    def __init__(self, nb_classes: int, pad_idx: int, eos_idx: int, label_smoothing_coeff=0.0):
        super(TransformerLoss, self).__init__()
        self.nb_classes = nb_classes
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing_coeff)

    # input: Tensor [sequence_size, batch_size, vocab_size] of linear output to give to softmax
    # target: Tensor [sequence_size, batch_size] of token indexes
    def forward(self, result, target) -> TransformerLossResult:
        flatten_result = result.view(-1, self.nb_classes)
        flatten_target = target.view(-1)
        loss_result = self.loss(flatten_result, flatten_target)
        return TransformerLossResult(loss_result, [])

    def partial_losses_dim(self) -> int:
        return 0
