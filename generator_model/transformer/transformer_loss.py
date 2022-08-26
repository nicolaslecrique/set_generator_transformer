import torch


class TransformerLossResult:

    # loss, avg on batch axis
    total_avg_loss: torch.Tensor

    # if loss is composed of several loss (multi-task learning), it contains loss for each sub-task
    partial_avg_losses: [torch.Tensor]

    def __init__(
            self,
            total_avg_loss: float,
            partial_avg_losses: [float]):

        self.total_avg_loss = total_avg_loss
        self.partial_avg_losses = partial_avg_losses


class TransformerLoss(torch.nn.Module):

    def __init__(self):
        super(TransformerLoss, self).__init__()

    def forward(self, result, target) -> TransformerLossResult:
        pass

    def partial_losses_dim(self) -> int:
        pass
