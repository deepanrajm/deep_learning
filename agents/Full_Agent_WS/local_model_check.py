## Simple Demo Program to use LM Studio / Ollama Local Host API

# ── Local Model ─────────────────────────────
API_KEY   = 'lm-studio'
BASE_URL  = 'http://localhost:1234/v1'
MODEL     = 'google/gemma-4-e4b'


# ── Import Packages ─────────────────────────────
from langchain_openai import ChatOpenAI


def get_llm():
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.1
    )

llm = get_llm()
print('LLM test:', llm.invoke('Say hello in one word.').content)