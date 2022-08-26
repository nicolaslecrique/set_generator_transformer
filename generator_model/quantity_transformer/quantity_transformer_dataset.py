from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from generator_model.data_loading.recipe_as_sequence_loader import RecipeAsSequence
from generator_model.transformer.transformer_batch import TransformerBatch


class RecipeWithQuantitiesDatasetItem:

    # array of tokens of size nb_components + 1 (sos token for meal type)
    token_sequence_with_sos_meal_token: np.ndarray
    # array of float of size nb_components
    ratios: np.ndarray
    normalized_weight: float

    def __init__(self,
                 token_sequence_with_sos_meal_token: np.ndarray,
                 ratios: np.ndarray,
                 normalized_weight: float):

        self.token_sequence_with_sos_meal_token = token_sequence_with_sos_meal_token
        self.ratios = ratios
        self.normalized_weight = normalized_weight


class RecipeWithQuantitiesDataset(Dataset):

    # max weight of a recipe in the whole dataset, for normalization
    max_weight: float
    recipes: [RecipeAsSequence]

    def __init__(self, recipes: [RecipeAsSequence], max_weight: float):
        self.recipes = recipes
        self.max_weight = max_weight

    def __getitem__(self, index) -> RecipeWithQuantitiesDatasetItem:

        recipe_as_sequence: RecipeAsSequence = self.recipes[index]

        choice_sos = np.random.choice(recipe_as_sequence.meal_courses_tokens)
        compo_with_sos = np.insert(recipe_as_sequence.components_tokens, 0, choice_sos)

        return RecipeWithQuantitiesDatasetItem(
            token_sequence_with_sos_meal_token=compo_with_sos,
            ratios=recipe_as_sequence.components_ratios,
            normalized_weight=recipe_as_sequence.weight / self.max_weight
        )

    def __len__(self):
        return len(self.recipes)


class QuantityTransformerTargetBatchForRatioLoss:

    # (sequence_size - 1 for sos_token, batch_size)
    ratios: torch.Tensor
    mask_not_padded: torch.Tensor

    def __init__(
            self,
            ratios: torch.Tensor,
            mask_not_padded: torch.Tensor):

        self.ratios = ratios
        self.mask_not_padded = mask_not_padded


class QuantityTransformerTargetBatchForWeightLoss:

    # (batch_size)
    weight: torch.Tensor

    def __init__(self, weight: torch.Tensor):
        self.weight = weight


class QuantityTransformerTargetBatch:

    target_for_ratios_loss: QuantityTransformerTargetBatchForRatioLoss
    target_for_weight_loss: QuantityTransformerTargetBatchForWeightLoss

    def __init__(
            self,
            target_for_ratios_loss: QuantityTransformerTargetBatchForRatioLoss,
            target_for_weight_loss: QuantityTransformerTargetBatchForWeightLoss
    ):
        self.target_for_ratios_loss = target_for_ratios_loss
        self.target_for_weight_loss = target_for_weight_loss


class CollateFctPadRecipeWithQuantities:

    def __init__(self, pad_token):
        self.pad_token = pad_token

    #  param is (compo_with_sos, ratios, weight)
    def __call__(self, batch: List[RecipeWithQuantitiesDatasetItem]) -> TransformerBatch:
        # size of the longest sequence
        target_size_input = len(max(batch, key=lambda recipe: len(recipe.token_sequence_with_sos_meal_token)).token_sequence_with_sos_meal_token)

        components_padded_list = []
        ratios_padded_list = []
        weight_list = []

        for recipe_sequence in batch:

            tokens = recipe_sequence.token_sequence_with_sos_meal_token
            ratios = recipe_sequence.ratios
            normalized_weight = recipe_sequence.normalized_weight

            current_size_input = len(tokens)

            components_padded = np.pad(
                tokens,
                (0, target_size_input - current_size_input),
                'constant',
                constant_values=(0, self.pad_token))

            ratios_padded = np.pad(
                ratios,
                (0, target_size_input - current_size_input),
                'constant',
                constant_values=(0, 999999.))

            components_padded_list.append(components_padded)
            ratios_padded_list.append(ratios_padded)
            weight_list.append(normalized_weight)

        components_batch = np.asarray(components_padded_list, dtype=np.int64)
        ratios_batch = np.asarray(ratios_padded_list, dtype=np.float32)
        weight_batch = np.asarray(weight_list, dtype=np.float32)

        tensor_data = torch.from_numpy(components_batch).t().contiguous()
        tensor_ratios = torch.from_numpy(ratios_batch).t().contiguous()
        tensor_weights = torch.from_numpy(weight_batch)

        tensor_mask = tensor_data.ne(self.pad_token)[1:, :]

        return TransformerBatch(
            model_input_tensor=tensor_data,
            target=QuantityTransformerTargetBatch(
                target_for_ratios_loss=QuantityTransformerTargetBatchForRatioLoss(tensor_ratios, tensor_mask),
                target_for_weight_loss=QuantityTransformerTargetBatchForWeightLoss(tensor_weights)
            )
        )
