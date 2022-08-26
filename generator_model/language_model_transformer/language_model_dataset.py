import numpy as np
import torch
from torch.utils.data import Dataset

from generator_model.data_loading.recipe_as_sequence_loader import RecipeAsSequence
from generator_model.transformer.transformer_batch import TransformerBatch


class RecipeDatasetAsLanguageModelItem:

    # int array of token_id: sequence with start token and end token
    model_input: np.ndarray
    # model_input shifted by one
    target: np.ndarray

    def __init__(self,
                 model_input: np.ndarray,
                 target: np.ndarray
                 ):
        self.model_input = model_input
        self.target = target


class RecipeDatasetAsLanguageModel(Dataset):

    eos_token: int
    recipes: [RecipeAsSequence]

    def __init__(
            self,
            recipes: [RecipeAsSequence],
            eos_token: int):
        self.recipes = recipes
        self.eos_token = eos_token

    def __getitem__(self, index) -> np.ndarray:

        recipe: RecipeAsSequence = self.recipes[index]

        sos_idxes = recipe.meal_courses_tokens
        sequence = recipe.components_tokens

        # start token is chosen among valid start token for current recipe
        # it's a case of conditional transformer (conditional on the dish type), similar to what is done in
        # https://arxiv.org/pdf/1909.05858.pdf
        sos_idx = np.random.choice(sos_idxes)
        # recipe components are shuffled (there is no intrinsic natural ordering)
        permuted_sequence = np.random.permutation(sequence)

        model_input = np.append(np.insert(permuted_sequence, 0, sos_idx), self.eos_token)
        # target is "next token", so it's input shifted left
        target = model_input[1:]

        return RecipeDatasetAsLanguageModelItem(
            model_input=model_input,
            target=target
        )

    def __len__(self):
        return len(self.recipes)


def pad_rows(rows: [np.ndarray], target_size: int, pad_token: int) -> np.ndarray:
    padded = [np.pad(row,(0, target_size - len(row)),'constant', constant_values=(0, pad_token)) for row in rows]
    padded_np = np.asarray(padded, dtype=np.int64)
    return padded_np


class CollateFctForRecipeAsLanguageModel:

    pad_token: int

    def __init__(
            self,
            pad_token: int):
        self.pad_token = pad_token

    def __call__(self, batch: [RecipeDatasetAsLanguageModelItem]) -> TransformerBatch:

        # pad inputs and targets then collate
        model_inputs = [item.model_input for item in batch]
        target = [item.target for item in batch]

        max_sequence_size = len(max(model_inputs, key=lambda row: len(row)))
        padded_model_input = pad_rows(model_inputs, max_sequence_size, self.pad_token)
        padded_target = pad_rows(target, max_sequence_size, self.pad_token)

        # transpose (batch_size, sequence_size) to (sequence_size, batch_size) and reorder dimensions in memory
        # to match the expected input format of Transformer model
        tensor_model_input = torch.from_numpy(padded_model_input).t().contiguous()
        tensor_target = torch.from_numpy(padded_target).t().contiguous()

        return TransformerBatch(
            model_input_tensor=tensor_model_input,
            target=tensor_target
        )
