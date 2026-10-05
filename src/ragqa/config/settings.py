"""环境配置（pydantic-settings）：API key、模型名、路径。

约定（与 .env.example 对应）：
- embedding / rerank 走 DashScope（DASHSCOPE_API_KEY + LLM_BASE_URL）
- 生成走 DeepSeek 官方（GEN_API_KEY / GEN_BASE_URL / GEN_MODEL_ID）；
  未设置时回退旧变量（LLM_*），便于过渡期兼容
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from ragqa.utils.paths import REPO_ROOT, resolve_path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── 路径 ──────────────────────────────────────────────
    data_dir: str = "data"
    papers_dir: str = "data/papers"
    upload_dir: str = "data/uploads"
    faiss_persist_dir: str = "faiss_db"

    # ── DashScope（embedding / rerank）────────────────────
    dashscope_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    embed_mode: Literal["api", "local"] = "api"  # 旧字段（=local 时优先于 provider）
    embed_provider: Literal["dashscope", "openai", "local"] = "dashscope"
    embed_api_model: str = "qwen3.7-text-embedding-flash"
    embed_dimensions: int = 0        # >0 时随请求发送；0 = 用模型原生维度（DashScope 内部回退 1024）
    embed_batch_size: int = 10
    embed_base_url: str = ""         # openai 兼容供应商（默认回退 llm_base_url）
    embed_api_key: str = ""          # 默认回退 dashscope_api_key
    embed_extra_body: str = ""       # JSON 字符串：供应商特有参数（如 {"text_type": "..."}）
    embed_local_model: str = "models/bge-m3"

    rerank_mode: Literal["api", "local"] = "api"
    rerank_api_model: str = "qwen3.7-text-rerank"
    rerank_local_model: str = "models/bge-reranker-base"

    # ── 生成（DeepSeek 官方；GEN_* 未设时回退 LLM_*）─────
    gen_api_key: str = ""
    gen_base_url: str = ""
    gen_model_id: str = ""
    llm_model_id: str = "qwen-max"  # 旧变量，仅作回退

    # ── 派生属性 ──────────────────────────────────────────
    @property
    def gen_key(self) -> str:
        return self.gen_api_key or self.dashscope_api_key

    @property
    def gen_url(self) -> str:
        return self.gen_base_url or self.llm_base_url

    @property
    def gen_model(self) -> str:
        return self.gen_model_id or self.llm_model_id

    @property
    def embed_provider_effective(self) -> str:
        return "local" if self.embed_mode == "local" else self.embed_provider

    @property
    def embed_model_name(self) -> str:
        """当前生效的嵌入模型名（进索引元信息 / trace。索引与模型强绑定）。"""
        return self.embed_local_model if self.embed_provider_effective == "local" else self.embed_api_model

    @property
    def rerank_model_name(self) -> str:
        return self.rerank_local_model if self.rerank_mode == "local" else self.rerank_api_model

    @property
    def faiss_dir(self) -> Path:
        return resolve_path(self.faiss_persist_dir)

    @property
    def upload_dir_path(self) -> Path:
        return resolve_path(self.upload_dir)

    @property
    def papers_dir_path(self) -> Path:
        return resolve_path(self.papers_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
