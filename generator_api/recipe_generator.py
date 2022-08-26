from typing import Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pydantic import BaseModel
import numpy as np

from generator_model.data_loading.recipe_as_sequence_loader import sos_tokens, eos_token
from generator_model.language_model_transformer.language_model_transformer import LanguageModelTransformer
from generator_model.language_model_transformer.sequence_generator import SequenceGenerator
from generator_model.quantity_transformer.quantity_transformer import QuantityTransformer, \
    QuantityTransformerBatchResult

@dataclass_json
@dataclass(frozen=True)
class GeneratedRecipe(BaseModel):
    meal_course: str
    components: Dict[str, float]


class RecipeGenerator:

    def __init__(
            self,
            language_model_transformer: LanguageModelTransformer,
            quantity_transformer: QuantityTransformer):

        language_model_transformer.eval()
        quantity_transformer.eval()

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
        self.quantity_transformer = quantity_transformer

    def generate_recipe(self):
        generated_idx: np.ndarray = self.sequence_generator.generate(15, 10, [])
        quantity_transformer_input = generated_idx[:-1].unsqueeze(-1)  # unsqueezed and without eos for qty transformer

        quantities_result: QuantityTransformerBatchResult = self.quantity_transformer.forward(quantity_transformer_input)
        weight_normalized = quantities_result.regression_output.squeeze()
        weight = weight_normalized * self.quantity_transformer.weight_normalization_factor
        linear_output = quantities_result.linear_output_for_softmax
        ratios = linear_output.softmax(dim=0).squeeze(1)
        weight_components = ratios * weight

        components_idx_list = generated_idx[1:-1].tolist()  # remove sos, eos
        components_weights_list = weight_components.tolist()

        components_with_qties = {self.idx_to_token[idx]: weight for idx, weight in zip(components_idx_list,components_weights_list)}
        meal_course= self.idx_to_token[generated_idx[0]]

        recipe = GeneratedRecipe(meal_course= meal_course, components=components_with_qties)
        return recipe
