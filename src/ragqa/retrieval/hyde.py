"""
HyDE (Hypothetical Document Embeddings)：先生成假设性答案，再用它做检索。
"""
from __future__ import annotations

from rag.generator import generate_simple

# HyDE 提示词：让 LLM 生成一段貌似真实文档的段落（假设性答案）
HYDE_PROMPT = (
    "请根据以下问题，撰写一段详细、专业的中文段落来回答该问题。"
    "要求：内容详实、包含具体技术细节、用第三人称客观陈述、"
    "仿佛这是一篇教科书或技术文档中的原文。不要出现'根据问题'等元话语。\n\n"
    "问题：{query}"
)


def generate_hypothetical_doc(query: str, max_tokens: int = 256) -> str:
    """
    用 LLM 生成一段假设性文档（HyDE）。
    这段文档不是最终答案，而是作为检索 query，去向量库中匹配真正相关的文档。
    """
    prompt = HYDE_PROMPT.format(query=query)
    try:
        hypo = generate_simple(prompt)
        if hypo and len(hypo.strip()) > 10:
            return hypo.strip()
        return query
    except Exception as e:
        # 如果 LLM 调用失败，fallback 原 query
        import logging
        logging.getLogger("rag").warning("[HyDE] 生成假设文档失败，回退原始 query: %s", e)
        return query
