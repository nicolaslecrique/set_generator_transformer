from typing import Set

import numpy as np

from generator_model.data_loading.recipes_loader import Recipe


def build_feature_matrix(
        recipes: [Recipe],
        with_components_features: bool = False,
        with_transformed_food_features: bool = True,
        with_nutriments_features: bool = False,
        with_linked_tags_features: bool = False,
        with_inferred_tags_features: bool = False) -> (np.ndarray, [str]):
    features: Set[str] = set()
    # 1) extract all features
    for recipe in recipes:
        if with_components_features:
            features |= set(component.uri for component in recipe.components)
        if with_transformed_food_features:
            features |= set(transformed_food.uri for component in recipe.components for transformed_food in component.transformed_foods )
        if with_nutriments_features:
            features |= set(nut.uri for nut in recipe.nutritional_values)
        if with_linked_tags_features:
            features |= set(tag.uri for tag in recipe.linked_tags)
        if with_inferred_tags_features:
            features |= set(tag.uri for tag in recipe.inferred_tags)
    features_list = list(features)
    features_list.sort()
    feature_to_index = {features_list[i]: i for i in range(0, len(features_list))}
    feature_matrix = np.zeros((len(recipes), len(features_list)), dtype=np.float32)
    for id_recipe, recipe in enumerate(recipes):
        if with_components_features:
            for c in recipe.components:
                feature_matrix[id_recipe, feature_to_index[c.uri]] = c.quantity_in_gram
        if with_transformed_food_features:
            for c in recipe.components:
                for tf in c.transformed_foods:
                    feature_matrix[id_recipe, feature_to_index[tf.uri]] = tf.quantity_in_gram
        if with_nutriments_features:
            for n in recipe.nutritional_values:
                feature_matrix[id_recipe, feature_to_index[n.uri]] = n.value
        if with_linked_tags_features:
            for t in recipe.linked_tags:
                feature_matrix[id_recipe, feature_to_index[t.uri]] = 1.
        if with_inferred_tags_features:
            for t in recipe.inferred_tags:
                feature_matrix[id_recipe, feature_to_index[t.uri]] = 1.

    return feature_matrix, features_list
