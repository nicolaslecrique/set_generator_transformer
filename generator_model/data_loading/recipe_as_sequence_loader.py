from typing import List

import numpy as np

from generator_model.data_loading.feature_matrix_builder import build_feature_matrix
from generator_model.data_loading.recipes_loader import Recipe, MealCourses


class RecipeAsSequence:

    recipe: Recipe
    weight: float
    components_tokens: [np.ndarray]
    components_quantities: [np.ndarray]
    components_ratios: [np.ndarray]
    meal_courses_tokens: [np.array]

    def __init__(
            self,
            recipe: Recipe,
            weight: float,
            components_tokens,
            components_quantities,
            components_ratios,
            meal_courses_tokens
    ):

        self.recipe = recipe
        self.weight = weight
        self.components_tokens = components_tokens
        self.components_quantities = components_quantities
        self.components_ratios = components_ratios
        self.meal_courses_tokens = meal_courses_tokens


class RecipesSetAsSequences:
    idx_to_token: [str]
    pad_idx: int
    eos_idx: int
    sos_idxes: [int]

    recipes: [RecipeAsSequence]

    def __init__(
            self, idx_to_token: [str],
            pad_idx: int,
            sos_idxes: [int],
            eos_idx: int,
            recipes: [RecipeAsSequence]
    ):
        self.recipes = recipes
        self.eos_idx = eos_idx
        self.sos_idxes = sos_idxes
        self.pad_idx = pad_idx
        self.idx_to_token = idx_to_token


sos_breakfast = '<sos_breakfast>'
sos_starter = '<sos_starter>'
sos_main_course = '<sos_main_course>'
sos_dessert = '<sos_dessert>'
sos_morning_snack = '<sos_morning_snack>'
sos_afternoon_snack = '<sos_afternoon_snack>'

eos_token = '<eos>'

sos_tokens = [
    sos_breakfast,
    sos_starter,
    sos_main_course,
    sos_dessert,
    sos_morning_snack,
    sos_afternoon_snack
]


def to_sos_tokens(meal_courses: MealCourses, idx_to_token: [str]) -> np.ndarray:
    meal_courses_token = []
    if meal_courses.breakfast:
        meal_courses_token.append(sos_breakfast)
    if meal_courses.starter:
        meal_courses_token.append(sos_starter)
    if meal_courses.main_course:
        meal_courses_token.append(sos_main_course)
    if meal_courses.dessert:
        meal_courses_token.append(sos_dessert)
    if meal_courses.morning_snack:
        meal_courses_token.append(sos_morning_snack)
    if meal_courses.afternoon_snack:
        meal_courses_token.append(sos_afternoon_snack)

    return np.array([idx_to_token.index(token) for token in meal_courses_token])


def load_recipes_as_sequences(recipes: [Recipe]) -> RecipesSetAsSequences:

    components_feature_matrix: np.ndarray
    features: List[str]
    components_feature_matrix, features = build_feature_matrix(recipes)

    total_weight_by_recipe = components_feature_matrix.sum(1, keepdims=True)
    components_feature_matrix_scaled_to_one = components_feature_matrix / total_weight_by_recipe

    # return un tuple of two lists where tuple_1 are recipe idx and tuple_2 are component idx
    non_zero_indexes_by_dim = np.nonzero(components_feature_matrix)

    special_tokens = ['<pad>'] + sos_tokens + [eos_token]
    pad_idx = special_tokens.index('<pad>')
    sos_idxes = [special_tokens.index(token) for token in sos_tokens]
    eos_idx = special_tokens.index(eos_token)

    idx_to_token = special_tokens + features

    recipes_tokens = [[] for _ in range(len(recipes))]
    recipes_quantities = [[] for _ in range(len(recipes))]
    recipes_ratios = [[] for _ in range(len(recipes))]

    for recipe_idx, component_idx in zip(*non_zero_indexes_by_dim):

        recipes_tokens[recipe_idx].append(component_idx + len(special_tokens))
        recipes_quantities[recipe_idx].append(components_feature_matrix[recipe_idx, component_idx])
        recipes_ratios[recipe_idx].append(
            components_feature_matrix_scaled_to_one[recipe_idx, component_idx])

    recipe_sequences = []

    for recipe, tokens, qties, ratios in zip(recipes, recipes_tokens, recipes_quantities, recipes_ratios):

        recipe_sequence = RecipeAsSequence(
            recipe=recipe,
            weight=sum(qties),
            components_tokens=np.asarray(tokens, dtype=np.int64),
            components_quantities=np.asarray(qties, dtype=np.float32),
            components_ratios=np.asarray(ratios, dtype=np.float32),
            meal_courses_tokens=to_sos_tokens(recipe.meal_courses, idx_to_token)
        )

        recipe_sequences.append(recipe_sequence)

    return RecipesSetAsSequences(
        recipes=recipe_sequences,
        idx_to_token=idx_to_token,
        pad_idx=pad_idx,
        sos_idxes=sos_idxes,
        eos_idx=eos_idx)


