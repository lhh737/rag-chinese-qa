"""
端点探针：实测 DashScope 三个端点（embedding / rerank / generation）的可用性与接口格式。

用途（设计 §3"P0 逐项安装冒烟验证"）：
- embedding：确认模型可用、实际向量维度、参数兼容性（dimensions / text_type）
- rerank：确认服务端点与响应格式（不同模型的 rerank 接口格式可能不同）
- generation：确认生成可用并记录 usage 结构（成本模型依据）

用法：PYTHONUTF8=1 python scripts/probe_endpoints.py
注意：只打印接口形状与维度，不打印 API Key。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMB_MODEL = os.environ.get("EMBED_API_MODEL", "text-embedding-v4")
RERANK_MODEL = os.environ.get("RERANK_API_MODEL", "gte-rerank-v2")
LLM_MODEL = os.environ.get("LLM_MODEL_ID", "qwen-max")

print(f"key_loaded={bool(KEY)} key_len={len(KEY)}")
print(f"models: embed={EMB_MODEL} rerank={RERANK_MODEL} llm={LLM_MODEL}")
print("-" * 70)

from openai import OpenAI

client = OpenAI(api_key=KEY, base_url=BASE)


# ── 1. embedding ─────────────────────────────────────────────

def try_embed(name: str, **kw) -> None:
    try:
        r = client.embeddings.create(model=EMB_MODEL, input=["这是一段用于探测向量维度的中文测试文本。"], **kw)
        v = r.data[0].embedding
        print(f"[embed:{name}] OK dim={len(v)} first3={[round(x, 5) for x in v[:3]]}")
        usage = getattr(r, "usage", None)
        print(f"[embed:{name}] usage={usage}")
    except Exception as e:
        msg = str(e).replace(KEY, "***")[:300] if KEY else str(e)[:300]
        print(f"[embed:{name}] FAIL {type(e).__name__}: {msg}")


try_embed("minimal")
try_embed("dims_1024", dimensions=1024)
try_embed("text_type", extra_body={"text_type": "query"})

print("-" * 70)

# ── 2. rerank ────────────────────────────────────────────────

import requests

DOCS = ["Transformer 的核心是自注意力机制。", "今天天气不错，适合出门散步。", "自注意力机制允许模型关注序列中的不同位置。"]


def try_rerank(name: str, url: str, payload: dict) -> None:
    try:
        r = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            timeout=30,
        )
        body = r.text.replace(KEY, "***")[:600]
        print(f"[rerank:{name}] status={r.status_code} resp={body}")
    except Exception as e:
        print(f"[rerank:{name}] FAIL {type(e).__name__}: {str(e)[:200]}")


# 候选 1：旧代码使用的服务化端点（gte-rerank 格式）
try_rerank(
    "svc_v1",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    {"model": RERANK_MODEL, "input": {"query": "自注意力机制", "documents": DOCS}, "parameters": {"top_n": 3, "return_documents": False}},
)
# 候选 2：兼容模式 rerank（OpenAI 风格）
try_rerank(
    "compat_v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/rerank",
    {"model": RERANK_MODEL, "query": "自注意力机制", "documents": DOCS, "top_n": 3},
)
# 候选 3：服务化端点去掉重复路径段
try_rerank(
    "svc_short",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank",
    {"model": RERANK_MODEL, "input": {"query": "自注意力机制", "documents": DOCS}, "parameters": {"top_n": 3, "return_documents": False}},
)

print("-" * 70)

# ── 3. generation ────────────────────────────────────────────

try:
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "只回复两个字：好的"}],
        max_tokens=16,
        temperature=0,
    )
    print(f"[gen] OK content={r.choices[0].message.content!r}")
    print(f"[gen] usage={r.usage}")
except Exception as e:
    msg = str(e).replace(KEY, "***")[:300] if KEY else str(e)[:300]
    print(f"[gen] FAIL {type(e).__name__}: {msg}")

print("-" * 70)
print("probe done")
