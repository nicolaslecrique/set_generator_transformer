import torch
from torch.nn import functional as F

from generator_model.transformer.transformer_loss import TransformerLoss, TransformerLossResult


class SoftCrossEntropyTransformerLossFromScratch(TransformerLoss):

    def __init__(self, nb_classes: int, pad_idx: int, eos_idx: int, label_smoothing_coeff=0.0):
        super(SoftCrossEntropyTransformerLossFromScratch, self).__init__()
        self.nb_classes = nb_classes
        self.label_smoothing_coeff = label_smoothing_coeff
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    # input: Tensor [sequence_size, batch_size, vocab_size] of linear output to give to softmax
    # target: Tensor [sequence_size, batch_size] of token indexes
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> TransformerLossResult:

        target_proba_smoothed = self._compute_unordered_probabilistic_target(target)

        # II) ---- COMPUTE LOSS ----

        # we flatten tensors on softmax dimension
        target_proba_smoothed_flattened = target_proba_smoothed.view(-1, self.nb_classes)
        target_flattened = target.view(-1)
        output_flattened = output.view(-1, self.nb_classes)

        # compute log_softmax
        log_prb_output = F.log_softmax(output_flattened, dim=1)

        # compute cross entropy by softmax
        loss_by_softmax = -(target_proba_smoothed_flattened * log_prb_output).sum(dim=1)

        # restrict loss computation to non padded elements
        non_pad_mask = target_flattened.ne(self.pad_idx)  # vector of True where target is not pad_idx
        loss_reduced_to_not_pad_index = loss_by_softmax.masked_select(non_pad_mask)

        return TransformerLossResult(loss_reduced_to_not_pad_index.mean(), [])


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

        # 4) apply label smoothing
        target_proba_smoothed = self._apply_label_smoothing_to_target(nb_targets, target_proba)

        return target_proba_smoothed

    def _apply_label_smoothing_to_target(self, nb_targets, target_proba):
        # https://arxiv.org/pdf/1512.00567.pdf, https://arxiv.org/abs/1906.02629
        nb_targets_smoothing = self.nb_classes - nb_targets
        smoothing_value = self.label_smoothing_coeff / nb_targets_smoothing
        target_proba.mul_(1.0 - self.label_smoothing_coeff)  # apply smoothing
        target_proba_smoothed = torch.max(target_proba, smoothing_value)  # use broadcasting
        return target_proba_smoothed

    def partial_losses_dim(self) -> int:
        return 0
