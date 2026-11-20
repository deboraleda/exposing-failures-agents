N_RUNS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 4096

# LLMs used by DeepEval as judges (provider must exist in PROVIDER_CONFIG).
# Every run is evaluated by all judges.
JUDGES = [
    {"provider": "deepinfra", "model": "Qwen/Qwen3.8-27B"},
    {"provider": "deepinfra", "model": "MiniMaxAI/MiniMax-M3"},
]

PROVIDER_CONFIG = {
    "openai": {
        "class": "ChatOpenAI",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        # gpt-6-luna only supports function tools + reasoning on /v1/responses,
        # and does not accept the temperature parameter
        "use_responses_api": True,
        "supports_temperature": False,
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
    "openai": ["gpt-6-luna"],
    "deepinfra": [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "zai-org/GLM-5.3-Flash",
    ],
}
