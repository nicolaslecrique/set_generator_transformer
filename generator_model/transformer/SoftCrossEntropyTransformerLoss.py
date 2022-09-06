import torch
from torch import nn as nn

from generator_model.transformer.transformer_loss import TransformerLoss, TransformerLossResult


class SoftCrossEntropyLossTransformerLoss(TransformerLoss):

    def __init__(self, nb_classes: int, pad_idx: int, eos_idx: int, label_smoothing_coeff=0.0):
        super(TransformerLoss, self).__init__()
        self.nb_classes = nb_classes
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing_coeff, ignore_index=pad_idx)

    # result: Tensor [sequence_size, batch_size, vocab_size] of linear output to give to softmax
    # target: Tensor [sequence_size, batch_size] of token indexes
    def forward(self, result, target) -> TransformerLossResult:

        # I) ---- PREPARE TARGET ----
        target_proba = self._compute_unordered_probabilistic_target(target)

        # II) ---- COMPUTE LOSS ----
        # we flatten tensors on softmax dimension
        flatten_result = result.view(-1, self.nb_classes)
        flatten_target = target_proba.view(-1, self.nb_classes)
        loss_result = self.loss(flatten_result, flatten_target)
        return TransformerLossResult(loss_result, [])

    def _compute_unordered_probabilistic_target(self, target):
        # 1) [sequence_size, batch_size] of indexes to [sequence_size, batch_size, vocabulary_ntokens] one_hot
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.nb_classes)
        # 2) valid next token is any token in the subsequent sequence except pad and eos
        seq_length = len(target)
        for i_seq_idx in range(seq_length - 2, -1, -1):
            # elt[i] = elt[i] + elt[i+1]
            to_paste = target_one_hot[i_seq_idx + 1, :, :].clone()
            to_paste[:, self.pad_idx] = 0
            to_paste[:, self.eos_idx] = 0

            target_one_hot[i_seq_idx, :, :].add_(to_paste)
        # 3) if there is several possible next token, give each one equals probability
        nb_targets = target_one_hot.sum(dim=2, dtype=torch.float, keepdims=True)  # keepdims to be broadcastable
        target_proba = target_one_hot.float()  # cast int to float
        target_proba.div_(nb_targets)
        return target_proba

    def partial_losses_dim(self) -> int:
        return 0
