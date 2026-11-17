N_RUNS = 1
TEMPERATURE = 0.7
MAX_TOKENS = 4096

# LLM used by DeepEval as judge (provider must exist in PROVIDER_CONFIG)
JUDGE_PROVIDER = "deepinfra"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

PROVIDER_CONFIG = {
    "openai": {
        "class": "ChatOpenAI",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepinfra": {
        "class": "ChatOpenAI",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_TOKEN",
    },
    # "anthropic": {
    #     "class": "ChatAnthropic",
    #     "base_url": None,
    #     "api_key_env": "ANTHROPIC_API_KEY",
    # },
}

MODELS = {
    #"openai": ["gpt-6-luna"], #, "gpt-4o-mini"],
    "deepinfra": ["Qwen/Qwen3-14B"]#, "Qwen/Qwen2.5-7B-Instruct",
    #               "meta-llama/Meta-Llama-3.1-8B-Instruct", # fraco, vai falhar mais
    #               ],
    #"anthropic": ["claude-3-5-sonnet-20240620"]
}