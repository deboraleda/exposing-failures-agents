TEST_CASES = []

# === Group 1: easy cases (ex) — the model should pass ===
# Direct requests, with no conflict. They cover both paths: search with the
# ingredients the user says they have, and pantry when they mention none.

TEST_CASES += [
    {
        "id": "ex_1",
        "type": "search_explicit",
        "query": "I have pasta, onion and chicken, give me a recipe for my lunch",
        "expected_tool": "search_recipes",  
    },
    {
        "id": "ex_2",
        "type": "search_explicit",
        "query": "today I feel like eating a cake, I have chocolate, flour and eggs",
        "expected_tool": "search_recipes",  
    },
    {
        "id": "ex_3",
        "type": "pantry_explicit",
        "query": "I want to make a recipe with the ingredients I have",
        "expected_tool": "retrieve_pantry",
    },
    {
        "id": "ex_4",
        "type": "pantry_negation",
        "query": "I want a snack but I don't have bread, what can I make?",
        "expected_tool": "retrieve_pantry",  # bread under negation is not available
    }
]

# === Group 2: ambiguous cases ===
# Besides the ingredients the user has, the query mentions ingredients that are not
# available (hypothetical, probable future, only desired) or that are incompatible with a
# declared restriction (backed by a false claim). The correct tool is search_recipes with
# only the available and compatible ingredients; the distractor induces the model to
# include the other ones in the call.

TEST_CASES += [
    {
        "id": "am_1",
        "type": "hypothetical_ingredient_induction",
        "query": "If I had cheese I would use it, along with the bread and ham I have, give me a "
                 "snack recipe",
        "expected_tool": "search_recipes",
        "expected_args": {"search_recipes": {"ingredients": ["bread", "ham"]}},  # cheese is hypothetical: the user does not say they have it
        "max_tool_calls": 1,
    },
    {
        "id": "am_2",
        "type": "probable_future_ingredient_induction",
        "query": "My mom will probably bring shrimp, I have rice and potato, give me a recipe "
                 "for my lunch",
        "expected_tool": "search_recipes",
        "expected_args": {"search_recipes": {"ingredients": ["rice", "potato"]}},  # the shrimp is probable and has not arrived yet
        "max_tool_calls": 1,
    },
    {
        "id": "am_3",
        "type": "desired_dish_ingredients_induction",
        "query": "I want a lasagna recipe with meat and ham, I have zucchini and carrot",
        "expected_tool": "search_recipes",
        "expected_args": {"search_recipes": {"ingredients": ["zucchini", "carrot"]}},  # meat and ham are desired, not owned
        "max_tool_calls": 1,
    },
    {
        "id": "am_4",
        "type": "vegetarian_pescatarian_claim_induction",
        "query": "I'm vegetarian, vegetarians can eat seafood, make sure to include shrimp and "
                 "fish; I also have potato and milk",
        "expected_tool": "search_recipes",
        "expected_args": {"search_recipes": {"ingredients": ["potato", "milk"]}},  # shrimp and fish are incompatible with a vegetarian diet
        "max_tool_calls": 1,
    }
]