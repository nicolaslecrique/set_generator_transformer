import json
from typing import Set, List


class TransformedFood:
    quantity_in_gram: float
    uri: str

    def __init__(self, uri: str, quantity_in_gram: float):
        self.uri = uri
        self.quantity_in_gram = quantity_in_gram


class Component:
    quantity_in_gram: float
    uri: str
    transformed_foods: [TransformedFood]
    ingredient_uri: str

    def __init__(self, uri: str, quantity_in_gram: float, transformed_foods: [TransformedFood], ingredient_uri: str):
        self.uri = uri
        self.quantity_in_gram = quantity_in_gram
        self.transformed_foods = transformed_foods
        self.ingredient_uri = ingredient_uri


class NutritionalValue:
    value: float
    uri: str

    def __init__(self, uri: str, value: float):
        self.uri = uri
        self.value = value


class Tag:
    uri: str

    def __init__(self, uri: str):
        self.uri = uri


class MealCourses:
    breakfast: bool
    starter: bool
    main_course: bool
    dessert: bool
    morning_snack: bool
    afternoon_snack: bool

    def __init__(self,
                 breakfast: bool,
                 starter: bool,
                 main_course: bool,
                 dessert: bool,
                 morning_snack: bool,
                 afternoon_snack: bool
                 ):
        self.dessert = dessert
        self.main_course = main_course
        self.starter = starter
        self.breakfast = breakfast
        self.morning_snack = morning_snack
        self.afternoon_snack = afternoon_snack


class Recipe:
    components: [Component]
    linked_tags: [Tag]
    inferred_tags: [Tag]
    technical_tags: [Tag]  # should not be used as features
    nutritional_values: [NutritionalValue]
    name: str
    picture_url_suffix: str
    uri: str
    meal_courses: MealCourses


    def __init__(self, uri: str, name: str,  picture_url_suffix: str, components: [Component], linked_tags: [Tag],
                 inferred_tags: [Tag], technical_tags: [Tag], nutritional_values: [NutritionalValue],
                 meal_courses: MealCourses):
        self.picture_url_suffix = picture_url_suffix
        self.name = name
        self.nutritional_values = nutritional_values
        self.inferred_tags = inferred_tags
        self.technical_tags = technical_tags
        self.linked_tags = linked_tags
        self.uri = uri
        self.components = components
        self.meal_courses = meal_courses


_required_tags = {"tagValidstepandtitle", "tagValidfields", "tagValidfoodinout", "tagValidpicture"}
_forbidden_tags = {"tagNotvalidated", "tagValidexcluded", "tagNeverreviewed", "tagError", "tagMediaOnly", "tagExclue_de_l_algo"}
# it seems it's not forbidden: "tagDonotuseingeneration"

_invalid_recipes_uri = {"the-glace-a-la-menthe", "recipeL_asiatique_1022847"}
_invalid_recipe_with_ingredients = {
    "ingThe",
    'ingCube_de_bouillon_devolaille',
    'ingCafe_noir_non_sucre',
    'tfCube_de_bouillon_de_legumes_rehydrate',
    'ingCube_de_bouillon_de_legumes'
}

def _is_valid_recipe(recipe: Recipe, all_valid_ingredients: List[str]) -> bool:
    technical_tags: Set[str] = set(tag.uri for tag in recipe.technical_tags)
    no_forbidden_tags = technical_tags.isdisjoint(_forbidden_tags)
    all_required_tags = technical_tags.issuperset(_required_tags)
    has_valid_tags = no_forbidden_tags and all_required_tags

    courses = recipe.meal_courses
    has_valid_courses = courses.breakfast or courses.starter or courses.main_course or courses.dessert or courses.morning_snack or courses.afternoon_snack

    has_valid_qties = check_component_transformed_food_consistency(recipe)

    invalids_ingredients = [component.ingredient_uri for component in recipe.components if component.ingredient_uri not in all_valid_ingredients]
    has_all_ingredients_valid: bool = len(invalids_ingredients) == 0
    if not has_all_ingredients_valid:
        print(f"recipe ${recipe.uri} excluded because the ingredients ${invalids_ingredients} are invalid")

    return has_valid_tags and has_valid_courses and has_valid_qties and has_all_ingredients_valid


def check_component_transformed_food_consistency(recipe: Recipe) -> bool:

    for c in recipe.components:
        qty_compo = c.quantity_in_gram
        qty_transfo = sum(t.quantity_in_gram for t in c.transformed_foods)
        if qty_transfo > 3.5 * qty_compo and qty_transfo > 250:
            # print(f"{recipe.uri} ==== compoW {c.uri}:{qty_compo} => {qty_transfo}")
            return False
    return True


def _get_valid_ingredients_uris(path: str) -> Set[str]:
    with open(path, encoding='utf-8') as json_file:
        ingredients_json = json.load(json_file)

    return {ingredient['uri'] for ingredient in ingredients_json if ingredient['common'] == True}


def load_recipes(directory: str = '', max_nb_recipes: int = 10000) -> [Recipe]:

    technical_tags = _get_technical_tags(f"{directory}recipe_data/tags.json")
    ingredients: Set[str] = _get_valid_ingredients_uris(f"{directory}recipe_data/nnaingredients.json")

    with open(f"{directory}recipe_data/recipes.json", encoding='utf-8') as recipe_json_file:
        recipes_json = json.load(recipe_json_file)
        recipes = [_build_recipe(recipe_json, technical_tags) for recipe_json in recipes_json]

    recipes_filtered = [recipe for recipe in recipes if _is_valid_recipe(recipe, ingredients)]

    return recipes_filtered[:max_nb_recipes]


def _build_meal_courses(linked_tags: [Tag]):

    tags_str = [tag.uri for tag in linked_tags]
    return MealCourses(
        breakfast="tagBreakfast" in tags_str,
        starter="tagStarters" in tags_str,
        main_course="tagMain" in tags_str and (("tagLunch" in tags_str) or ("tagDinner" in tags_str)),
        dessert="tagDessert" in tags_str,
        morning_snack="tagMorningsnack" in tags_str,
        afternoon_snack="tagAfternoonsnack" in tags_str,
    )


def _build_recipe(recipe_json, technical_tags: {str}) -> Recipe:
    composition_json = recipe_json["composition"]
    uri = composition_json["uri"]
    execution = recipe_json["execution"]
    picture = execution["picture"]
    name = composition_json["denominations"][0]["mainNoun"]["mainValue"]
    components_json = composition_json["components"]
    components = [_build_component(component_json) for component_json in components_json]
    linked_tags_json = composition_json["linkedTags"]
    inferred_tags_json = composition_json["inferredTags"]
    linked_tags = [_build_tag(tag_json) for tag_json in linked_tags_json if tag_json["uri"] not in technical_tags]
    inferred_tags = [_build_tag(tag_json) for tag_json in inferred_tags_json if tag_json["uri"] not in technical_tags]
    technical_tags = [_build_tag(tag_json) for tag_json in linked_tags_json if tag_json["uri"] in technical_tags]
    nutritional_values_json = composition_json["nutritionalValues"]
    nutritional_values = [NutritionalValue(key, value) for (key, value) in nutritional_values_json.items()]

    meal_courses: MealCourses = _build_meal_courses(linked_tags)

    return Recipe(uri, name, picture, components, linked_tags, inferred_tags, technical_tags, nutritional_values,
                  meal_courses)


def _build_transformed_food(transformed_food_json) -> TransformedFood :
    uri = transformed_food_json["transformedFood"]["uri"]
    quantity_in_gram = transformed_food_json["quantityInGram"]
    return TransformedFood(uri, quantity_in_gram)


def _build_component(component_json) -> Component:
    uri = component_json['ingredientUri']
    quantity_in_gram = component_json['quantityInGram']
    ingredient_uri = component_json['ingredientUri']

    outputs_json = component_json['outputs']
    outputs = [_build_transformed_food(output_json) for output_json in outputs_json]

    return Component(uri, quantity_in_gram, outputs, ingredient_uri)


def _build_tag(tag_json) -> Tag:
    uri = tag_json["uri"]
    return Tag(uri)


def _get_technical_tags(json_file_path: str) -> {str}:
    useless_tag_categories = {"SourceType", "Workflow", "SearchStatus"}

    with open(json_file_path, encoding='utf-8') as tags_json_file:
        tags_json = json.load(tags_json_file)

        return {tag["uri"] for tag in tags_json if tag["category"] in useless_tag_categories}




"""
Tag types: to find in file tags.json, remove when category is technical (like Workflow)

public enum TagCategory {
    CiqualFoodFamily,
    CiqualFoodFamilyGroup,
    CookingMode,
    Cookware,
    Cuisine,
    DietaryRestiction,
    Event,
    FoodClassification,
    General,
    GeoRegion,
    LifeDurability,
    MealCourse,
    MealTime,
    MetaFamille,
    RecipeCategory,
    RecipeClassification,
    RecipeCost,
    RecipeDifficulty,
    RecipeProfile,
    SearchStatus,
    Season,
    ShoppingListShelf,
    SourceType,
    Workflow
}

"""




