import math
import time
from typing import Callable

import torch
from torch import nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from generator_model.transformer.transformer_loss import TransformerLossResult, TransformerLoss
from generator_model.transformer.transformer_batch import TransformerBatch


def train_model_one_epoch(
        seq_to_seq_model: nn.Module,
        data_batch_to_nb_softmax: Callable[[torch.Tensor], int],
        train_data_loader: DataLoader,
        optimizer: Optimizer,
        criterion: TransformerLoss,
        epoch_index: int,
        scheduler: StepLR,
        nb_batches: int
):
    seq_to_seq_model.train()  # Turn on the train mode

    total_loss = 0.
    total_nb_softmax = 0

    start_time = time.time()
    batch: TransformerBatch
    for batch_idx, batch in enumerate(train_data_loader):

        # 1. initialize gradients
        optimizer.zero_grad()
        # 2. forward pass
        output = seq_to_seq_model(batch.model_input_tensor)
        # 3. compute loss
        loss_result: TransformerLossResult = criterion(output, batch.target)
        batch_loss = loss_result.total_avg_loss
        # 4. backward pass
        batch_loss.backward()
        # 5. gradient clipping, improve performance, cf. https://arxiv.org/pdf/1905.11881.pdf
        torch.nn.utils.clip_grad_norm_(seq_to_seq_model.parameters(), 0.5)
        # 6. update model parameters
        optimizer.step()

        # -------------- PRINT TRAINING INFO ---------------
        data_this_batch = data_batch_to_nb_softmax(batch.model_input_tensor)
        total_loss += batch_loss.item() * data_this_batch
        total_nb_softmax += data_this_batch
        log_interval = 10
        if batch_idx % log_interval == 0 and batch_idx > 0:

            # FOR DEBUG print_sequence(data, idx_to_token, 10)

            cur_loss = total_loss / total_nb_softmax
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.3f} | ms/batch {:5.2f} | '
                  'loss {:5.3f} | ppl {:8.2f}'.format(
                epoch_index, batch_idx, nb_batches, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(min(10, cur_loss))))
            start_time = time.time()
        # -------------- END PRINT TRAINING INFO ---------------


class ComparisonPrinter:

    def __call__(self, model_output: torch.Tensor, batch: TransformerBatch):
        pass


# evaluate model loss for a given set of model/data/loss
def evaluate(seq_to_seq_model: nn.Module,
             data_source: DataLoader,
             data_batch_to_nb_softmax: Callable[[torch.Tensor], int],
             criterion: TransformerLoss,
             print_comparison: ComparisonPrinter = None
             ) -> TransformerLossResult:

    seq_to_seq_model.eval()  # Turn on the evaluation mode

    total_loss = 0.
    partial_losses = [0] * criterion.partial_losses_dim()

    nb_predictions = 0  # batch_size * sequence length * nb_batch
    batch_printed = False
    with torch.no_grad():
        batch: TransformerBatch
        for batch_index, batch in enumerate(data_source):
            output = seq_to_seq_model(batch.model_input_tensor)
            if print_comparison is not None and not batch_printed:
                batch_printed = True
                print_comparison(output, batch)

            # criterion return average of loss (unity is on class)
            nb_elements_this_batch = data_batch_to_nb_softmax(batch.model_input_tensor)
            batch_loss: TransformerLossResult = criterion(output, batch.target)
            total_loss += nb_elements_this_batch * batch_loss.total_avg_loss
            nb_predictions += nb_elements_this_batch
            for partial_loss_idx in range(len(partial_losses)):
                partial_losses[partial_loss_idx] += nb_elements_this_batch * batch_loss.partial_avg_losses[partial_loss_idx]

    for partial_loss_idx in range(len(partial_losses)):
        partial_losses[partial_loss_idx] = partial_losses[partial_loss_idx] / (nb_predictions - 1)

    return TransformerLossResult(
        total_avg_loss=total_loss / (nb_predictions - 1),
        partial_avg_losses=partial_losses)


# global training of model
# evaluate model loss on validation set after each training epoch
def train_model(seq_to_seq_model: nn.Module,
                train_data_loader: DataLoader,
                validation_data_loader: DataLoader,
                optimizer: Optimizer,
                data_batch_to_nb_softmax: Callable[[torch.Tensor], int],
                train_criterion: TransformerLoss,
                eval_criterion: TransformerLoss,
                scheduler: StepLR,
                nb_batches_by_epoch: int,
                nb_epochs: int):

    for epoch_index in range(1, nb_epochs + 1):
        epoch_start_time = time.time()
        train_model_one_epoch(seq_to_seq_model, data_batch_to_nb_softmax, train_data_loader, optimizer, train_criterion,
                              epoch_index, scheduler, nb_batches_by_epoch)

        loss: TransformerLossResult = evaluate(seq_to_seq_model, validation_data_loader, data_batch_to_nb_softmax, eval_criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.3f}s | valid loss {:5.3f} | '
              'valid ppl {:8.2f}'.format(epoch_index, (time.time() - epoch_start_time),
                                         loss.total_avg_loss, math.exp(min(10,loss.total_avg_loss))))

        for partial_loss_idx in range(len(loss.partial_avg_losses)):
            print(f"partial loss {partial_loss_idx}: {loss.partial_avg_losses[partial_loss_idx]}")

        print('-' * 89)

        scheduler.step()
