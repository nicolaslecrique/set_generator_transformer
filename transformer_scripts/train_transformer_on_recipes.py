import math

import random
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from generator_model.transformer.transformer_loss import TransformerLossResult
from generator_model.language_model_transformer.language_model_loss import CrossEntropyUnorderedSequenceLoss
from generator_model.data_loading.recipe_as_sequence_loader import RecipesSetAsSequences, load_recipes_as_sequences, \
    RecipeAsSequence
from generator_model.language_model_transformer.language_model_dataset import RecipeDatasetAsLanguageModel, \
    CollateFctForRecipeAsLanguageModel
from generator_model.transformer.transformer_model import TransformerModel
from generator_model.language_model_transformer.sequence_generator import SequenceGenerator
from generator_model.transformer.transformer_training import evaluate, train_model
from generator_model.data_loading.recipes_loader import Recipe, load_recipes

# ================= prepare data ===================

torch.manual_seed(643532254)
np.random.seed(468472542)
random.seed(514742545)

# shuffle to get unbiased distribution of recipes between train / valid / test sets
recipes_not_shuffled = load_recipes('../data/')
recipes: [Recipe] = random.sample(recipes_not_shuffled, k=len(recipes_not_shuffled))
recipe_set_as_sequences: RecipesSetAsSequences = load_recipes_as_sequences(recipes)
idx_to_token = recipe_set_as_sequences.idx_to_token
pad_idx = recipe_set_as_sequences.pad_idx
eos_idx = recipe_set_as_sequences.eos_idx
sos_idxes = recipe_set_as_sequences.sos_idxes

recipes_as_sequences: [RecipeAsSequence] = recipe_set_as_sequences.recipes

nb_recipes: int = len(recipes)
start_valid_set_index = int(nb_recipes * 0.85)
start_test_set_index = int(nb_recipes * 0.95)

train_set_np = recipes_as_sequences[:start_valid_set_index]
valid_set_np = recipes_as_sequences[start_valid_set_index:start_test_set_index]
test_set_np = recipes_as_sequences[start_test_set_index:]

# ================= Setup model ===================

embedding_dim = 60  # embedding dimension
hidden_dim = 60  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # the number of heads in the multi head attention models
dropout = 0.05  # the dropout value
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(recipe_set_as_sequences.idx_to_token, embedding_dim, nhead, hidden_dim, nlayers, pad_idx, mask_next_tokens=True, positioning=False, dropout=dropout).to(device)

# ================= Setup learning config ===================


def nb_softmax(data: torch.Tensor) -> int:
    return (data != pad_idx).sum().item()

train_criterion = CrossEntropyUnorderedSequenceLoss(nb_classes=len(idx_to_token), pad_idx=pad_idx, eos_idx=eos_idx, label_smoothing_coeff=0.05)
eval_criterion = CrossEntropyUnorderedSequenceLoss(nb_classes=len(idx_to_token), pad_idx=pad_idx, eos_idx=eos_idx)

lr = 5.0  # learning rate
optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr)
scheduler: StepLR = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

batch_size = 50  # original : 20
eval_batch_size = 10
nb_epochs = 40  # The number of epochs, (was 3 on tutorial)

# ================= build train/valid/tes set ===================

train_dataset = RecipeDatasetAsLanguageModel(train_set_np, eos_token=eos_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateFctForRecipeAsLanguageModel(pad_idx))

valid_dataset = RecipeDatasetAsLanguageModel(valid_set_np, eos_token=eos_idx)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=True, collate_fn=CollateFctForRecipeAsLanguageModel(pad_idx))

test_dataset = RecipeDatasetAsLanguageModel(test_set_np, eos_token=eos_idx)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True, collate_fn=CollateFctForRecipeAsLanguageModel(pad_idx))

# ================== Train, evaluate, and generate samples ===============

nb_batches = len(train_dataset) // batch_size

train_model(model, train_loader, valid_loader, optimizer, nb_softmax, train_criterion, eval_criterion, scheduler, nb_batches, nb_epochs)
recipe_name = f'../model_save/transformer_recipe_{embedding_dim}emb_{hidden_dim}hid_{nlayers}lay_{dropout}drop_{lr}lr_{batch_size}batch_{nb_epochs}epo.pt'

torch.save(model, recipe_name)


test_loss: TransformerLossResult = evaluate(model, test_loader, nb_softmax, eval_criterion)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss.total_avg_loss, math.exp(test_loss.total_avg_loss)))

print('=' * 89)


generator = SequenceGenerator(model, sos_idxes, eos_idx, len(idx_to_token))
max_nb_ingredients = 20

for i in range(50):
    result = generator.generate(max_nb_ingredients, 20, [])
    result_as_id_list = result.numpy().tolist()
    result_token_list = [idx_to_token[idx] for idx in result_as_id_list]
    print(result_token_list)


# Current results:
# * test loss 4.29
# * valid loss 4.34
# * train loss 4.24


# TODO IDEES
# ===== Generation ========
# Ne pas répeter les ingrédients
# Top_k
# top_n (nucleus)
# temperature
# Beam-search
# cf. https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb
# cf. https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277

# === Training ====
# Label smoothing: Done
# Cross entropy contre tous les suivants possibles (sorte de cross entropy non ordonnée pour une sequence): Done
# regarder si on peut pas prendre en compte les caracteristiques des ingredients dans les embedding
# Adam
# Recherche hyper-parameters avec grid search ou random
# Idée ordre: classer de façon absolue les ingredients en mettant en 1er les plus discriminant:
# discriminant ? matrice ingredient / ingredient avec la proportion de recettes qui contienne l'ingrédient 2
# ensuite on classe par E[X^2]: un aliment dans toutes les recettes sera à la fin, un ingrédient dans aucune recette sera au début
