# llm_service.py
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from tool_use_failures.config import N_RUNS, TEMPERATURE, MAX_TOKENS, PROVIDER_CONFIG, MODELS
from dotenv import load_dotenv

load_dotenv(override=True)


def get_llm(target):
    match = None

    for provider, model_list in MODELS.items():
        for model_name in model_list:
            if target in model_name.lower():
                match = (provider, model_name)
                break
        if match:
            break

    if not match:
        available = [m for models in MODELS.values() for m in models]
        raise SystemExit(f"Model '{target}' not found in config. Available: {available}")

    provider, model_name = match
    print(f"\n=== Running {provider}/{model_name} ({N_RUNS} runs per case) ===")

    # ── Instantiate LLM ──────────────────────────────────────────────────────────
    config = PROVIDER_CONFIG[provider]

    if config["class"] == "ChatOpenAI":
        optional_kwargs = {}

        if config.get("supports_temperature", True):
            optional_kwargs["temperature"] = TEMPERATURE

        llm = ChatOpenAI(
            model=model_name,
            base_url=config["base_url"],
            api_key=os.getenv(config["api_key_env"]),
            max_tokens=MAX_TOKENS,
            use_responses_api=config.get("use_responses_api", False),
            **optional_kwargs,
        )
    elif config["class"] == "ChatAnthropic":
        llm = ChatAnthropic(
            model=model_name,
            api_key=os.getenv(config["api_key_env"]),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    return llm, provider, model_name