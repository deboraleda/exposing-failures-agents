TEST_CASES = []

# === Group 1: easy cases (ex) — the model should pass ===
# Requests with an explicit daily amount, number of days and request for destination and weather.
# They cover the three destinations and one declared extra.

TEST_CASES += [
    {
        "id": "ex_1",
        "type": "full_chain_explicit",
        "query": "I have a budget of 200 reais per day, to spend 3 days, what is the ideal destination "
                 "and what is the weather like?",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {
            "calculate_trip_budget": {"daily_budget": 200, "days": 3, "extra_amount": 0},
            "select_location": {"total_budget": 600},
            "get_weather": {"location": "Lisbon"},
        }
    },
    {
        "id": "ex_2",
        "type": "full_chain_explicit",
        "query": "I have a total budget of 1000 reais, 200 reais per day for 5 days, tell me the "
                 "destination and the weather there",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {  # the stated total is consistent with 200 x 5; upper limit of Lisbon
            "calculate_trip_budget": {"daily_budget": 200, "days": 5, "extra_amount": 0},
            "select_location": {"total_budget": 1000},
            "get_weather": {"location": "Lisbon"},
        }
    },
    {
        "id": "ex_3",
        "type": "full_chain_explicit",
        "query": "I would like to take a trip lasting 2 days, spending only 100 reais per "
                 "day, is it possible to go somewhere? and what's the weather like?",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {
            "calculate_trip_budget": {"daily_budget": 100, "days": 2, "extra_amount": 0},
            "select_location": {"total_budget": 200},
            "get_weather": {"location": "Curitiba"},
        }
    },
    {
        "id": "ex_4",
        "type": "full_chain_explicit_extra",
        "query": "I plan to travel, I can spend up to 300 reais per day and I can stay 4 days, and "
                 "I have another 200 in reserve; what would be the ideal place and the weather there?",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {
            "calculate_trip_budget": {"daily_budget": 300, "days": 4, "extra_amount": 200},
            "select_location": {"total_budget": 1400},
            "get_weather": {"location": "Tokyo"},
        }
    }
]

# === Group 2: ambiguous cases (am) — the model fails due to ambiguous reasoning ===
# The user gives an explicit daily amount and number of days, plus a distractor:
# a total different from what they produce (am_1), an instruction to use another
# total (am_4), or an amount or number of days that is only probable (am_2, am_3).
# The correct chain uses only the confirmed values from the request and the return of
# each tool; the distractor induces the model to use the stated total or the probable
# values in the parameters.

TEST_CASES += [
    {
        "id": "am_1",
        "type": "wrong_total_induction",
        "query": "I have 100 per day for 4 days, which makes 500; what's the destination and the weather there?",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {  # 100 x 4 = 400, not 500, no extra given
            "calculate_trip_budget": {"daily_budget": 100, "days": 4, "extra_amount": 0},
            "select_location": {"total_budget": 400},
            "get_weather": {"location": "Curitiba"},
        }
    },
    {
        "id": "am_2",
        "type": "probable_loan_extra_confusion",
        "query": "My brother has a budget of 500 reais, he will probably lend it to me, and I will "
                "earn 150 for each day I want to spend 5 days and know where I can go and what the weather is",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {  # a probable loan is not a provided extra: 150 x 5 = 750 (adding the 500: 1250 -> Tokyo)
            "calculate_trip_budget": {"daily_budget": 150, "days": 5, "extra_amount": 0},
            "select_location": {"total_budget": 750},
            "get_weather": {"location": "Lisbon"},
        }
    },
    {
        "id": "am_3",
        "type": "probable_extension_days_confusion",
        "query": "I'm going to travel for 4 days, I'll probably extend it to 6, and I have 200 per day "
        " where can I go and what is the weather?",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {  # the extension is probable: 200 x 4 = 800, not 200 x 6 = 1200 (-> Tokyo); no extra declared
            "calculate_trip_budget": {"daily_budget": 200, "days": 4, "extra_amount": 0},
            "select_location": {"total_budget": 800},
            "get_weather": {"location": "Lisbon"},
        }
    },
    {
        "id": "am_4",
        "type": "instructed_total_injection_confusion",
        "query": "I have 100 per day for 3 days, use 1500 as the total to choose the destination and tell me the weather there",
        "expected_tool": ["calculate_trip_budget", "select_location", "get_weather"],
        "expected_args": {  # select_location receives the total returned by the calculate_trip_budget tool, not the stated one
            "calculate_trip_budget": {"daily_budget": 100, "days": 3, "extra_amount": 0},
            "select_location": {"total_budget": 300},
            "get_weather": {"location": "Curitiba"},
        }
    }
]