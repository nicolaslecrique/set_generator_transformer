import math
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from generator_model.quantity_transformer.quantity_transformer_dataset import QuantityTransformerTargetBatch, \
    RecipeWithQuantitiesDataset, CollateFctPadRecipeWithQuantities
from generator_model.transformer.transformer_loss import TransformerLossResult
from generator_model.quantity_transformer.quantity_transformer_loss import QuantityTransformerLoss
from generator_model.data_loading.recipe_as_sequence_loader import RecipesSetAsSequences, load_recipes_as_sequences
from generator_model.data_loading.recipes_loader import Recipe, load_recipes
from generator_model.transformer.transformer_batch import TransformerBatch
from generator_model.transformer.transformer_model import TransformerModel
from generator_model.quantity_transformer.quantity_transformer import QuantityTransformerBatchResult, \
    QuantityTransformer
from generator_model.transformer.transformer_training import evaluate, train_model, ComparisonPrinter

# ================= prepare data ===================

torch.manual_seed(242542)
np.random.seed(9825425)
random.seed(41645645)

recipes: [Recipe] = load_recipes('../data/')
recipes_set_as_sequences: RecipesSetAsSequences = load_recipes_as_sequences(recipes)
idx_to_token = recipes_set_as_sequences.idx_to_token
pad_idx = recipes_set_as_sequences.pad_idx

recipes_as_sequences = recipes_set_as_sequences.recipes
# shuffle to get unbiased distribution of recipes between train / valid / test sets
random.shuffle(recipes_as_sequences)

max_weight = max(recipe.weight for recipe in recipes_as_sequences)

nb_recipes: int = len(recipes_as_sequences)
start_valid_set_index = int(nb_recipes * 0.75)
start_test_set_index = int(nb_recipes * 0.90)

train_set = recipes_as_sequences[:start_valid_set_index]
valid_set = recipes_as_sequences[start_valid_set_index:start_test_set_index]
test_set = recipes_as_sequences[start_test_set_index:]

# ================= Setup model ===================

embedding_dim = 15  # embedding dimension
hidden_dim = 15  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 3  # the number of heads in the multi head attention models
dropout = 0.0  # the dropout value
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nb_tokens = len(idx_to_token)
transformer_model = TransformerModel(nb_tokens, embedding_dim, nhead, hidden_dim, nlayers, pad_idx, False, positioning=False, dropout=dropout).to(device)
model = QuantityTransformer(transformer_model, pad_idx, recipes_set_as_sequences.idx_to_token, max_weight)

# ================= Setup learning config ===================

# one softmax by element of the batch: nb_softmax = batch_size = size(1)
def nb_softmax(data: torch.Tensor) -> int:
    return data.size(1)


class QuantityTransformerComparisonPrinter(ComparisonPrinter):

    def __init__(self, idx_to_token: [str]):
        self.idx_to_token = idx_to_token

    def __call__(self, model_output: QuantityTransformerBatchResult, batch: TransformerBatch):

        input_tensor = batch.model_input_tensor
        target: QuantityTransformerTargetBatch = batch.target

        nb_batches = input_tensor.size(1)

        output_softmax = model_output.linear_output_for_softmax
        output_qty = model_output.regression_output

        target_qty = target.target_for_weight_loss.weight
        mask = target.target_for_ratios_loss.mask_not_padded

        target_ratios = target.target_for_ratios_loss.ratios

        result_padded = output_softmax.masked_fill(mask.logical_not(), float('-inf'))
        result_softmax = result_padded.softmax(dim=0)

        for idx in range(nb_batches):
            components_idx_list = input_tensor[1:, idx].numpy().tolist()
            result_ratios_list = result_softmax[:, idx].numpy().tolist()
            target_ratios_list = target_ratios[:, idx].numpy().tolist()

            print("-----------------")
            print(f"POID: {target_qty[idx].item()} <> {output_qty[idx].item()}")
            for comp_idx, target_val, result_val in zip(components_idx_list, target_ratios_list, result_ratios_list):
                print(f'{self.idx_to_token[comp_idx]}: {target_val} <> {result_val}')



criterion = QuantityTransformerLoss(
    padding_token_idx=pad_idx,
    ratios_loss_coeff=0.7,
    total_weight_mse_loss_scaling=200.0
)


lr = 1.0  # learning rate
optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr)
scheduler: StepLR = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

batch_size = 50  # original : 20

eval_batch_size = 10
nb_epochs = 100  # The number of epochs, (was 3 on tutorial)

# ================= build train/valid/tes set ===================

train_dataset = RecipeWithQuantitiesDataset(train_set, max_weight)
valid_dataset = RecipeWithQuantitiesDataset(valid_set, max_weight)
test_dataset = RecipeWithQuantitiesDataset(test_set, max_weight)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=CollateFctPadRecipeWithQuantities(pad_idx))

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=True,
                                           collate_fn=CollateFctPadRecipeWithQuantities(pad_idx))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True,
                                          collate_fn=CollateFctPadRecipeWithQuantities(pad_idx))

# ================== Train, evaluate, and generate samples ===============

nb_batches = len(train_dataset) // batch_size

train_model(model, train_loader, valid_loader, optimizer, nb_softmax, criterion, criterion, scheduler, nb_batches, nb_epochs)

recipe_name = f'../model_save/transformer_qty_recipe_{embedding_dim}emb_{hidden_dim}hid_{nlayers}lay_{dropout}drop_{lr}lr_{batch_size}batch_{nb_epochs}epo.pt'

torch.save(model, recipe_name)


test_loss_result: TransformerLossResult = evaluate(model, test_loader, nb_softmax, criterion, QuantityTransformerComparisonPrinter(idx_to_token))
test_loss = test_loss_result.total_avg_loss
partial_losses = test_loss_result.partial_avg_losses

print('=' * 89)
print('| End of training | test loss {:5.3f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(min(10, test_loss))))

for partial_loss_idx in range(len(partial_losses)):
    print(f"partial loss {partial_loss_idx}: {partial_losses[partial_loss_idx]}")


print('=' * 89)

# Current results:
# * test loss 1.284 (loss0=1.359, loss1=1.107)
# * valid loss 1.291 (loss0=1.427, loss1=0.971)
# * train loss 1.156
