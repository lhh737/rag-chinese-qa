"""YAML 配置加载：一切可对照的参数进这里（retrieval / generation / pricing）。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ragqa.utils.paths import resolve_path

CONFIG_DIR = resolve_path(os.getenv("RAG_CONFIG_DIR", "config"))


class HydeConfig(BaseModel):
    enabled: bool = True
    max_tokens: int = 256
    temperature: float = 0.0
    enable_thinking: bool = False  # 混合思考模型默认关闭思考


class RetrievalConfig(BaseModel):
    """检索与分块参数（对照实验的全部自变量都应有对应字段）。"""

    # 分块
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 150
    child_chunk_size: int = 400
    child_chunk_overlap: int = 40
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""]
    )
    # 召回与融合
    vector_fetch_k: int = 24
    bm25_fetch_k: int = 24
    fusion: Literal["rrf", "merge"] = "rrf"
    rrf_k: int = 60
    # 重排与最终上下文
    rerank_top_n: int = 12
    base_context_k: int = 4
    max_context_k: int = 8
    # HyDE
    hyde: HydeConfig = Field(default_factory=HydeConfig)

    def config_id(self) -> str:
        """可对照配置的短标识（进 trace/Registry）。"""
        return (
            f"vec{int(self.vector_fetch_k)}_bm25{int(self.bm25_fetch_k)}_"
            f"{self.fusion}{self.rrf_k}_rr{int(self.rerank_top_n)}_"
            f"k{int(self.base_context_k)}-{int(self.max_context_k)}_"
            f"hyde{int(self.hyde.enabled)}"
        )


class GenerationConfig(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 1024
    prompt_version: str = "v1"
    # 混合思考模型（deepseek-flash / qwen3.x）：默认关闭思考，快且省
    enable_thinking: bool = False


class PricingConfig(BaseModel):
    """成本估算单价（元 / 1M tokens）。仅影响成本口径，不影响指标。

    数据来源（2026-10 快照，以官方价目为准）：
    - qwen3.7-text-embedding-flash: 0.125 元/M（百炼：0.000125 元/千 token）
    - deepseek-flash: 峰谷计价（闲时 输入1/输出4；高峰 输入2/输出8），缓存命中输入 0.02/0.04
    - qwen3.7-text-rerank: 未查到公开价，用 placeholder，待核实
    """

    embed: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {"qwen3.7-text-embedding-flash": {"input": 0.125}}
    )
    generation: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "deepseek-flash": {
                "input_idle": 1.0,
                "input_peak": 2.0,
                "output_idle": 4.0,
                "output_peak": 8.0,
                "cache_hit_idle": 0.02,
                "cache_hit_peak": 0.04,
            },
            "_default": {"input_idle": 1.0, "input_peak": 1.0, "output_idle": 4.0, "output_peak": 4.0},
        }
    )
    rerank: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {"qwen3.7-text-rerank": {"input": 0.8}, "_default": {"input": 0.8}}
    )


def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(**_load_yaml("retrieval.yml"))


@lru_cache(maxsize=1)
def load_generation_config() -> GenerationConfig:
    return GenerationConfig(**_load_yaml("generation.yml"))


@lru_cache(maxsize=1)
def load_pricing_config() -> PricingConfig:
    return PricingConfig(**_load_yaml("pricing.yml"))
