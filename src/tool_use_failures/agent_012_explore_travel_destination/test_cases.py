TEST_CASES = []

# === Group 1: easy cases (ex) — the model should pass ===
# Requests with a single intent (current weather or news/safety) and an identifiable destination.
# Exactly one tool must be called.

TEST_CASES += [
    {
        "id": "ex_1",
        "type": "weather_explicit",
        "query": "I'm traveling to Athens today, what's the weather like there?",
        "expected_tool": ["get_weather"],
        "max_tool_calls": 1,
    },
    {
        "id": "ex_2",
        "type": "news_explicit",
        "query": "I'm planning a trip to Argentina, I want to know if there is any safety risk",
        "expected_tool": ["search_destination_news"],
        "max_tool_calls": 1
    },
    {
        "id": "ex_3",
        "type": "weather_explicit",
        "query": "I want to travel to Orlando, but first tell me if it's sunny there",
        "expected_tool": ["get_weather"],
        "max_tool_calls": 1,
    },
    {
        "id": "ex_4",
        "type": "news_explicit",
        "query": "What is happening in Paris this week?",
        "expected_tool": ["search_destination_news"],
        "max_tool_calls": 1,
    }
]

# === Group 2: ambiguous cases (am) — the model fails due to ambiguous reasoning ===
# The user asks for current weather and news/safety, with insistence, explicit
# order or dependency between the steps. The rule in the prompt
# "Each execution may invoke at most one tool." allows only ONE of the two tools,
# the request induces the model to call both.
# The expected tool is the one for the intent the user prioritizes; when no priority
# is stated, it is the one for the intent that appears first in the request.
# Calling both is the failure (max_tool_calls = 1).
# am_4 has a single intent (current weather) phrased with a news keyword, which
# induces the model to call search_destination_news.

TEST_CASES += [
    {
        "id": "am_1",
        "type": "multi_intent_conflict",
        "query": "I need both pieces of information to decide on my trip: the weather in Lisbon and the "
                 "safety advisories, don't leave either one out",
        "expected_tool": ["get_weather"],  # intent mentioned first
        "max_tool_calls": 1,
    },
    {
        "id": "am_2",
        "type": "multi_intent_conflict",
        "query": "Check what the weather is like in New York, and based on that search whether there is any "
                 "outdoor event this week",
        "expected_tool": ["get_weather"],  # intent mentioned first
        "max_tool_calls": 1,
    },
    {
        "id": "am_3",
        "type": "stated_priority_vs_first_intent",
        "query": "I'm anxious to know the weather in São Paulo, but it's better to know whether, because of the "
                "current weather, there is any safety alert",
        "expected_tool": ["search_destination_news"],  # the priority is the alert, get_weather does not report alerts
        "max_tool_calls": 1,
    },
    {
        "id": "am_4",
        "type": "news_keyword_weather_intent",
        "query": "Is there any good news about the weather in Recife for today?",
        "expected_tool": ["get_weather"],  # good news about the weather = current weather
        "max_tool_calls": 1,
    }
]