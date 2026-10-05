"""本地模型适配（optional extra：sentence-transformers；无 GPU 默认不用）。

接口与 API 版保持一致（鸭子类型），便于配置切换：
- LocalBGEEmbedding.embed_documents / embed_query
- LocalCrossEncoderReranker.rerank -> [{"index", "relevance_score"}]
"""
from __future__ import annotations

from pathlib import Path

from ragqa.utils.logging import logger


class LocalBGEEmbedding:
    def __init__(self, model_path: str | Path, device: str = "cpu", batch_size: int = 4) -> None:
        from sentence_transformers import SentenceTransformer  # 懒导入（optional extra）

        path = str(model_path)
        if not Path(path).is_dir():
            raise FileNotFoundError(f"本地嵌入模型不存在: {path}（需下载 BGE 模型或改用 API 模式）")
        logger.info("[local] 嵌入模型: %s", path)
        self._model = SentenceTransformer(path, device=device)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class LocalCrossEncoderReranker:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder  # 懒导入（optional extra）

        path = str(model_path)
        if not Path(path).is_dir():
            raise FileNotFoundError(f"本地重排模型不存在: {path}（需下载 BGE 模型或改用 API 模式）")
        logger.info("[local] 重排模型: %s", path)
        self._model = CrossEncoder(path, device=device)
        self.last_usage: dict | None = None
        self.last_latency_ms: float = 0.0

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        pairs = [[query, d] for d in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
        return [{"index": i, "relevance_score": float(scores[i])} for i in ranked[: (top_n or len(documents))]]
