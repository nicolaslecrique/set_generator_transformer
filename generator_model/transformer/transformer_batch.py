import torch


class TransformerBatch:

    # tensor (sequence_size, batch_size, vocab_size)
    model_input_tensor: torch.Tensor
    target: object

    def __init__(
            self,
            model_input_tensor: torch.Tensor,
            target: object
    ):
        self.model_input_tensor = model_input_tensor
        self.target = target
