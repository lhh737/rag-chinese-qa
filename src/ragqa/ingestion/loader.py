"""
文档加载与分块：支持普通分块和父子分块两种策略。
"""
import hashlib
import os
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from utils.config_handler import faiss_conf


def compute_md5(filepath: str) -> str:
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_file(filepath: str) -> list[Document]:
    if filepath.endswith(".txt"):
        return TextLoader(filepath, encoding="utf-8").load()
    elif filepath.endswith(".pdf"):
        return PyPDFLoader(filepath).load()
    else:
        return []


def split_documents(documents: list[Document]) -> list[Document]:
    """普通分块（用于简单 RAG 模式）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split(filepath: str) -> list[Document]:
    """加载文件并普通分块。"""
    docs = load_file(filepath)
    if not docs:
        return []
    filename = os.path.basename(filepath)
    for d in docs:
        d.metadata["source"] = filename
    return split_documents(docs)


def load_and_split_parent_child(filepath: str) -> tuple[list[Document], list[Document]]:
    """
    父子分块：大块（父块）保留完整上下文，小块（子块）用于检索。
    返回 (parent_docs, child_docs)，子块 metadata 中 parent_id 指向父块。
    """
    docs = load_file(filepath)
    if not docs:
        return [], []

    filename = os.path.basename(filepath)
    for d in docs:
        d.metadata["source"] = filename

    # 父块切分器（大块，保留上下文）
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(faiss_conf.get("parent_chunk_size", 1200)),
        chunk_overlap=int(faiss_conf.get("parent_chunk_overlap", 150)),
        separators=faiss_conf.get("separators", ["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""]),
    )

    # 子块切分器（小块，用于检索）
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(faiss_conf.get("child_chunk_size", 400)),
        chunk_overlap=int(faiss_conf.get("child_chunk_overlap", 40)),
        separators=faiss_conf.get("separators", ["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""]),
    )

    # 先切父块
    parent_docs = parent_splitter.split_documents(docs)
    # 为每个父块生成 parent_id
    import uuid
    child_docs = []
    for p_doc in parent_docs:
        pid = str(uuid.uuid4())
        p_doc.metadata["parent_id"] = pid
        # 从父块再切子块
        children = child_splitter.split_documents([p_doc])
        for c in children:
            c.metadata["parent_id"] = pid
            c.metadata["source"] = filename
        child_docs.extend(children)

    return parent_docs, child_docs


def get_file_info(filepath: str) -> dict:
    stat = os.stat(filepath)
    return {
        "filename": os.path.basename(filepath),
        "path": filepath,
        "size_kb": round(stat.st_size / 1024, 1),
        "md5": compute_md5(filepath),
        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
