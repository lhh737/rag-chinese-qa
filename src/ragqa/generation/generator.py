from openai import OpenAI
from config.settings import LLM_MODEL_ID, DASHSCOPE_API_KEY, LLM_BASE_URL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=LLM_BASE_URL)
    return _client


def generate(prompt: str, context: str = "", question: str = "") -> str:
    system_msg = prompt
    if context:
        system_msg = f"{prompt}\n\n参考以下文档内容回答问题：\n{context}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question or "请根据提供的文档内容回答问题"},
    ]

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def generate_simple(question: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content