
TEST_CASES = []

# ============================================================================
# Group 1: Basic rejection and scope handling
# ============================================================================

TEST_CASES += [
    {
        "id": "nt_1",
        "type": "empty_input",
        "query": "",
        "acceptable_tools": [],
        "expected_tool_count": 0,
        "expected_args": None,
        "expected_status": "failure",
        "expected_response": "Error: The user request cannot be empty.",
    },
    {
        "id": "nt_2",
        "type": "unrelated",
        "query": "What's the weather going to be like in Campina Grande tomorrow?",
        "acceptable_tools": [],
        "expected_tool_count": 0,
        "expected_args": None,
        "expected_status": "success",
        "expected_response": "Query outside of academic scope.",
    },
]

# ============================================================================
# Group 2: Normal requests — baseline controls
# ============================================================================

TEST_CASES += [
    {
        "id": "fd_1",
        "type": "broad_academic_request",
        "query": "Find academic papers about artificial intelligence.",
        "acceptable_tools": ["search_arxiv"],
        "expected_tool_count": 1,
        "expected_args": {"research_query": "artificial intelligence"},
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "fd_2",
        "type": "explicit_life_sciences_search",
        "query": "Find academic papers about Alzheimer's disease.",
        "acceptable_tools": ["search_pubmed"],
        "expected_tool_count": 1,
        "expected_args": {"research_query": "Alzheimer's disease"},
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "st_1",
        "type": "concept_explanation",
        "query": "Explain what reinforcement learning is.",
        "acceptable_tools": ["get_wikipedia_summary"],
        "expected_tool_count": 1,
        "expected_args": {"research_query": "reinforcement learning"},
        "expected_status": "success",
        "expected_response": None,
    },
]

# ============================================================================
# Group 3: Ambiguity and domain boundaries
# ============================================================================

TEST_CASES += [
    {
        "id": "am_1",
        "type": "implicit_concept_vs_research",
        "query": (
            "I'm preparing for an exam on molecular chemistry. "
            "I understand the basic terminology, but I'm struggling to "
            "understand how molecular orbital theory explains chemical "
            "bonding. I would also like to explore what researchers have "
            "been investigating about this topic."
        ),
        "acceptable_tools": [
            "get_wikipedia_summary",
            "search_arxiv"
        ],
        "expected_tool_count": 1,
        "expected_args": {
            "research_query": "molecular orbital theory and chemical bonding"
        },
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_2",
        "type": "implicit_research_vs_concept",
        "query": (
            "I've been reading about Alzheimer's disease for a class project. "
            "I already know the basic definition, but I want to understand "
            "what is currently known about the mechanisms behind its "
            "progression and how scientists are investigating them."
        ),
        "acceptable_tools": [
            "search_pubmed",
            "get_wikipedia_summary"
        ],
        "expected_tool_count": 1,
        "expected_args": {
            "research_query": "mechanisms of Alzheimer's disease progression"
        },
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_3",
        "type": "implicit_domain_classification",
        "query": (
            "For my research project, I'm investigating how neural networks "
            "can help identify mutations associated with inherited diseases. "
            "I need to understand the scientific approaches used in this area "
            "and what has been published about their applications."
        ),
        "acceptable_tools": [
            "search_arxiv",
            "search_pubmed"
        ],
        "expected_tool_count": 1,
        "expected_args": {
            "research_query": (
                "neural networks for identifying mutations associated "
                "with inherited diseases"
            )
        },
        "expected_status": "success",
        "expected_response": None,
    },
    {
        "id": "am_4",
        "type": "implicit_multiple_domains",
        "query": (
            "I'm preparing a literature review covering two topics: "
            "autonomous vehicle navigation and the molecular mechanisms "
            "of antibiotic resistance. I need an overview of the research "
            "available in both areas."
        ),
        "acceptable_tools": [],
        "expected_tool_count": 0,
        "expected_args": None,
        "expected_status": "success",
        "expected_response": (
            "The query spans multiple distinct domains. "
            "Please make one specific research request at a time."
        ),
    },
]
