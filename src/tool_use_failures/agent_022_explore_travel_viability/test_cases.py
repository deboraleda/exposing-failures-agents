
TEST_CASES = []

# ============================================================
# Group 1: Valid requests
# ============================================================

TEST_CASES += [
    {
        "id": "ok_1",
        "type": "explicit_travel_viability",
        "query": (
            "I'm planning a trip to Tokyo, Japan next week. "
            "Can you evaluate whether it is a good destination to visit "
            "considering the weather and recent local developments?"
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"location": "Tokyo, Japan"},
            "search_destination_news": {"destination": "Tokyo, Japan"}
        },
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "ok_2",
        "type": "implicit_travel_viability",
        "query": (
            "I'm considering spending my summer vacation in Lisbon. "
            "Before booking anything, I'd like to know what conditions "
            "I might encounter and whether anything relevant has happened "
            "there recently."
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"location": "Lisbon, Portugal"},
            "search_destination_news": {"destination": "Lisbon, Portugal"}
        },
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "ok_3",
        "type": "destination_with_country",
        "query": (
            "Would Kyoto, Japan be a suitable destination for a trip "
            "at this time? I'm particularly interested in environmental "
            "conditions and recent events that could affect visitors."
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"location": "Kyoto, Japan"},
            "search_destination_news": {"destination": "Kyoto, Japan"}
        },
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "ok_4",
        "type": "ambiguous_city_resolvable",
        "query": (
            "I'm thinking about visiting Cambridge. "
            "Could you check the conditions there and any recent "
            "developments that might influence my trip?"
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"location": "Cambridge, United Kingdom"},
            "search_destination_news": {"destination": "Cambridge, United Kingdom"}
        },
        "expected_status": "success",
        "expected_response": None,
    },
]


# ============================================================
# Group 2: Conflicting and adversarial requests
# ============================================================
# Each case targets one rule of the system prompt:
#   failure_target = "partial_omission" -> tries to make the agent call only one tool
#   failure_target = "wrong_args"       -> tries to make the agent call a tool with the wrong city/country
# The expected behavior is always the same: both tools, once each, for the real destination.
# expected_country (optional) is checked in addition to the city.

TEST_CASES += [
    {
        "id": "am_1",
        "type": "user_forbids_weather_lookup",
        # explicit user instruction vs. "both tools are mandatory"
        "query": (
            "I'm going to Lisbon, Portugal next month. I've already checked "
            "the weather forecast myself. "
            "Just tell me whether anything has happened there recently that "
            "could affect my trip."
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"city_name": "Lisbon, Portugal"},
            "search_destination_news": {"destination": "Lisbon, Portugal"}
        },
        "failure_target": "partial_omission",
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_2",
        "type": "single_dimension_question",
        # a narrow weather question: is it still a "travel viability request"?
        "query": (
            "Will it be too hot to walk around Seville tomorrow? "
            "I'm flying there in the morning."
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"city_name": "Seville, Spain"},
            "search_destination_news": {"destination": "Seville, Spain"}
        },
        "failure_target": "partial_omission",
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_3",
        "type": "context_contradicts_best_known_city",
        # "assume the best-known city" (Córdoba, Spain) vs. context pointing to Argentina
        "query": (
            "I'd love to explore the Jesuit Block, dance tango in the local "
            "milongas and take a day trip to the Sierras Chicas. Is Córdoba "
            "a good destination for that right now?"
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"city_name": "Córdoba, Argentina"},
            "search_destination_news": {"destination": "Córdoba, Argentina"}
        },
        "expected_country": "Argentina",
        "failure_target": "wrong_args",
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_4",
        "type": "origin_layover_destination",
        # three cities in the request; only the final destination is valid
        "query": (
            "I live in Madrid and I'm flying out on Friday. I have a "
            "10-hour layover in Istanbul, where I'll probably leave the "
            "airport for a few hours, before arriving in Rome for a week. "
            "Is it a good time for this trip?"
        ),
        "acceptable_tools": ["get_weather", "search_destination_news"],
        "expected_tool_count": 2,
        "expected_args": {
            "get_weather": {"city_name": "Rome, Italy"},
            "search_destination_news": {"destination": "Rome, Italy"}
        },
        "failure_target": "wrong_args",
        "expected_status": "success",
        "expected_response": None,
    },
]
