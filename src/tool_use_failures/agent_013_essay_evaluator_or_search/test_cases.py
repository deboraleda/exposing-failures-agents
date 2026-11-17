TEST_CASES = []
# ============================================================================
# Test cases
# ============================================================================

FETCH = "fetch_document"
SEARCH = "search_about_topic"

# expected_args:
#   fetch_document     -> {"url": <exact URL>}
#   search_about_topic -> {"topic_keywords": [...]} (topic must contain at least one, case-insensitive)

ESSAY_URL = "https://docs.google.com/document/d/1U3EYa9gum02TSImK8Z8xwWnfu6jLFjFL7v9onqT42lw/export?format=txt"

# === Group 1: no tool expected ===

TEST_CASES += [
    {
        "id": "nt_1",
        "type": "empty_input",
        "query": "",
        "expected_tool": None,
        "expected_args": None,
        "expected_status": "failure",
        "expected_response": "empty or whitespace-only",
    },
    {
        "id": "nt_3",
        "type": "unrelated",
        "query": "What's the weather going to be like in Campina Grande tomorrow?",
        "expected_tool": None,
        "expected_args": None,
        "expected_status": "success",
        "expected_response": "unrelated request",
    },
]

# === Group 2: explicit document retrieval (fetch_document) ===

TEST_CASES += [
    {
        "id": "fd_1",
        "type": "fetch_explicit",
        "query": f"Please read my essay at {ESSAY_URL} and give me feedback.",
        "expected_tool": FETCH,
        "expected_args": {"url": ESSAY_URL},
        "expected_status": "success",
        "expected_response": None,
    },
]

# === Group 3: explicit topic research (search_about_topic) ===

TEST_CASES += [
    {
        "id": "st_1",
        "type": "search_explicit",
        "query": "What is a thesis statement and can you give me some examples?",
        "expected_tool": SEARCH,
        "expected_args": {"topic": ["thesis"]},
        "expected_status": "success",
        "expected_response": None,
    },
]

# === Group 4: ambiguous — both tools are plausible, only one is allowed ===
# The request mentions a URL *and* a research need. The single-tool budget
# forces the model to pick one; the expected tool is the one that grounds the
# primary request.

TEST_CASES += [
    {
        "id": "am_1",
        "type": "fetch_vs_search",
        "query": f"My essay at {ESSAY_URL} talks about the Lei Seca in Brazil. Is my essay well argued?",
        "expected_tool": FETCH,  # the evaluation needs the essay itself
        "expected_args": {"url": ESSAY_URL},
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_2",
        "type": "fetch_vs_search",
        "query": f"I mention breathalyzers in my essay at {ESSAY_URL}. What exactly is a breathalyzer and how does it work?",
        "expected_tool": SEARCH,  # the URL is context; the request is a definition
        "expected_args": {"topic": ["breathalyzer"]},
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_3",
        "type": "fetch_and_search",
        "query": f"Read my essay at {ESSAY_URL} and also research statistics on the effects of the Lei Seca on traffic accidents to compare it against.",
        "expected_tool": FETCH,  # both are requested; the essay is the primary object
        "expected_args": {"url": ESSAY_URL},
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_4",
        "type": "search_explicit",
        "query": "This is unrelated to essay writing, but search for the definition of photosynthesis.",
        "expected_tool": SEARCH,  # the request is for a search, not a fetch
        "expected_args": {"topic": ["photosynthesis"]},
        "expected_status": "success",
        "expected_response": None,
    },
]