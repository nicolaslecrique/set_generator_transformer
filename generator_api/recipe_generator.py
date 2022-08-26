from typing import Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pydantic import BaseModel
import numpy as np

from generator_model.data_loading.recipe_as_sequence_loader import sos_tokens, eos_token
from generator_model.language_model_transformer.language_model_transformer import ModelTransformer
from generator_model.language_model_transformer.sequence_generator import SequenceGenerator


@dataclass_json
@dataclass(frozen=True)
class GeneratedRecipe(BaseModel):
    meal_course: str
    components: set[str]


class RecipeGenerator:

    def __init__(
            self,
            language_model_transformer: ModelTransformer):

        language_model_transformer.eval()

        self.idx_to_token = language_model_transformer.idx_to_token
        token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        sos_token_idxes = [token_to_idx[token] for token in sos_tokens]
        eos_token_idx = token_to_idx[eos_token]
        nb_tokens = len(token_to_idx)

        sequence_generator = SequenceGenerator(
            language_model_transformer,
            sos_token_idxes,
            eos_token_idx,
            nb_tokens)

        self.sequence_generator = sequence_generator

    def generate_recipe(self):
        generated_idx: np.ndarray = self.sequence_generator.generate(15, 10, [])


        components_idx_list = generated_idx[1:-1].tolist()  # remove sos, eos

        components = {self.idx_to_token[idx] for idx in components_idx_list}
        meal_course= self.idx_to_token[generated_idx[0]]

        recipe = GeneratedRecipe(meal_course= meal_course, components=components)
        return recipe
