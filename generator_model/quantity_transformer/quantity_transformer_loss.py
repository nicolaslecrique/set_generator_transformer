import torch
from torch import nn as nn
from torch.nn import functional as F, MSELoss

from generator_model.quantity_transformer.quantity_transformer_dataset import QuantityTransformerTargetBatch, \
    QuantityTransformerTargetBatchForRatioLoss
from generator_model.transformer.transformer_loss import TransformerLoss, TransformerLossResult
from generator_model.quantity_transformer.quantity_transformer import QuantityTransformerBatchResult


# compute soft cross entropy loss for quantity ratio: https://arxiv.org/pdf/1708.00584.pdf
class SoftCrossEntropyLossWithPadding(torch.nn.Module):

    pad_idx: int

    def __init__(self, pad_idx: int):
        super(SoftCrossEntropyLossWithPadding, self).__init__()
        self.pad_idx = pad_idx

    # result: tensor (nb_components, batch_size)
    def forward(self, result: torch.FloatTensor, target: QuantityTransformerTargetBatchForRatioLoss):

        target_qty = target.ratios
        mask = target.mask_not_padded

        mask_padding = mask.logical_not()

        result_padded = result.masked_fill(mask_padding, float('-inf'))

        result_softmax = F.log_softmax(result_padded, dim=0)
        loss_by_item = - result_softmax * target_qty
        loss_by_item_zeroed_on_padding = loss_by_item.masked_fill(mask_padding, 0.0)  # padd loss
        loss_by_sequence = loss_by_item_zeroed_on_padding.sum(dim=0)  # compute sum on non padding elements
        mean_loss = loss_by_sequence.mean()  # average on batches
        return mean_loss


class QuantityTransformerLoss(TransformerLoss):

    # coeff to scale MSE to same order of magnitude as Cross entropy
    total_weight_mse_loss_scaling: float
    # coeff to adjust importance of ratio loss relative to total weight MSE loss
    ratios_loss_coeff: float

    # loss for MSE total weight
    criterion_quantity: MSELoss
    # loss for ratios
    criterion_proportions: SoftCrossEntropyLossWithPadding

    def __init__(
            self,
            padding_token_idx: int,
            ratios_loss_coeff: float,
            total_weight_mse_loss_scaling: float
    ):
        super(QuantityTransformerLoss, self).__init__()

        self.criterion_proportions = SoftCrossEntropyLossWithPadding(pad_idx=padding_token_idx)
        self.criterion_quantity = nn.MSELoss()
        self.ratios_loss_coeff = ratios_loss_coeff
        self.total_weight_mse_loss_scaling = total_weight_mse_loss_scaling

    def forward(
            self,
            result: QuantityTransformerBatchResult,
            target: QuantityTransformerTargetBatch
    ) -> TransformerLossResult:

        softmax_loss = self.criterion_proportions(
            result=result.linear_output_for_softmax,
            target=target.target_for_ratios_loss
        )

        mse_loss = self.criterion_quantity(
            input=result.regression_output,
            target=target.target_for_weight_loss.weight
        )

        scaled_mse_loss = self.total_weight_mse_loss_scaling * mse_loss

        total_loss = self.ratios_loss_coeff * softmax_loss + (1 - self.ratios_loss_coeff) * scaled_mse_loss
        return TransformerLossResult(total_loss, [softmax_loss, scaled_mse_loss])

    def partial_losses_dim(self) -> int:
        return 2


