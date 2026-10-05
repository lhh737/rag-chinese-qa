"""文档解析：pypdf / txt 直连（替代 langchain PyPDFLoader / TextLoader，设计 §5.4）。

- PDF：逐页一个 Document（metadata: doc_id / source / page）
- doc_id = 文件名去扩展名（论文即 arXiv ID；与评测集 supporting_docs 的口径一致）
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader

from ragqa.types import Document
from ragqa.utils.logging import logger

SUPPORTED_EXTS = (".pdf", ".txt")


def doc_id_of(filepath: str | Path) -> str:
    return Path(filepath).stem


def compute_content_hash(filepath: str | Path) -> str:
    """文件内容 sha256（摄取幂等键：doc_id + content_hash + pipeline_version）。"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    did = doc_id_of(path)
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # 单页抽取失败不拖垮整篇
            logger.warning("[loader] %s 第 %d 页抽取失败: %s", path.name, i, e)
            text = ""
        docs.append(Document(page_content=text, metadata={"doc_id": did, "source": path.name, "page": i}))
    return docs


def _load_txt(path: Path) -> list[Document]:
    did = doc_id_of(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("[loader] %s 非 UTF-8，回退 GBK 解码", path.name)
        text = path.read_text(encoding="gbk", errors="replace")
    return [Document(page_content=text, metadata={"doc_id": did, "source": path.name, "page": 0})]


def load_file(filepath: str | Path) -> list[Document]:
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".txt":
        return _load_txt(path)
    raise ValueError(f"不支持的文件类型: {ext}（仅支持 {SUPPORTED_EXTS}）")
