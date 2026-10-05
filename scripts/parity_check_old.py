"""
依赖迁移对照实验 · old phase —— 旧栈检索基线导出。

⚠️ 本脚本依赖迁移前的旧代码（rag/、utils/、model/）。如需复现，先在迁移前提交
（5af6e15 或更早）的工作区运行；迁移后请使用 scripts/parity_check_new.py 做对照。

背景（设计 §5.4 通过标准）：依赖迁移完成后，需验证"迁移前后同 5 条查询的检索结果一致"。
本脚本在**迁移前**、用**旧代码 + 全局环境**（已装 langchain 全家桶）跑一遍检索，
导出到 eval_results/parity/old_stack.json，供迁移后的 new phase 对照。

运行（从仓库根目录）：
    PYTHONUTF8=1 python scripts/parity_check_old.py

导出内容：
- 10 篇文档（8 论文 + 2 中文 txt）的父/子块边界（内容 sha1 + 长度 + 头部 60 字）
- 5 条查询的 ① 向量 top-k（sha1+距离）② BM25 top-k ③ 最终上下文父块（sha1）
- 零宽字符/控制符计数（评估后续"文档清洗"对切分的影响）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# 必须在导入 config.settings 之前设置：让旧栈索引写入临时目录，避免污染正式 faiss_db
PARITY_TMP = os.path.join(REPO_ROOT, ".parity")
os.environ["FAISS_PERSIST_DIR"] = os.path.join(PARITY_TMP, "old_index")

from utils.config_handler import faiss_conf
from utils.path_tool import get_abs_path

# ─────────────────────────────────────────────────────────────
# 对照实验固定输入（与 new phase 保持一致，修改需两边同步）
# ─────────────────────────────────────────────────────────────

DOC_FILES = [
    "data/papers/1706.03762.pdf",
    "data/papers/2005.11401.pdf",
    "data/papers/2005.14165.pdf",
    "data/papers/2204.05862.pdf",
    "data/papers/2205.01068.pdf",
    "data/papers/2302.13971.pdf",
    "data/papers/2303.08774.pdf",
    "data/papers/2312.10997.pdf",
    "data/llm_intro.txt",
    "data/test_transformer.txt",
]

QUERIES = [
    "Transformer 的注意力机制计算复杂度是多少？",
    "RAG 是什么？它解决了什么问题？",
    "GPT-3 模型有多少参数？",
    "LLaMA 模型的训练数据规模是多少？",
    "什么是大语言模型？",
]

# 零宽字符 + 控制符码点区间（文档清洗的剥离对象；用码点判断避免源码中出现不可见字符）
_DIRTY_RANGES = (
    (0x200B, 0x200F),  # 零宽空格/连接符/方向标记
    (0x202A, 0x202E),  # 双向控制符
    (0x2060, 0x2060),  # word joiner
    (0xFEFF, 0xFEFF),  # BOM / 零宽不换行空格
    (0x0000, 0x0008),  # C0 控制符（保留 \t\n\r）
    (0x000B, 0x000C),
    (0x000E, 0x001F),
)


def count_dirty_chars(text: str) -> int:
    n = 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _DIRTY_RANGES):
            n += 1
    return n


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def doc_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def head(text: str, n: int = 60) -> str:
    return text[:n].replace("\n", " ")


def main() -> None:
    os.makedirs(PARITY_TMP, exist_ok=True)

    from rag.document_loader import get_file_info, load_and_split_parent_child
    from rag.hybrid_retriever import HybridRetriever
    from rag.vector_store import VectorStore

    out: dict = {
        "phase": "old",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "retrieval": dict(faiss_conf),
            "note": "hyde=off（对照只比较原始 query 的检索链路）",
        },
        "documents": {},
        "queries": [],
    }

    print("[parity-old] loading documents + building temp index ...")
    vs = VectorStore()
    for rel in DOC_FILES:
        path = get_abs_path(rel)
        did = doc_id(path)
        t0 = time.time()
        parent_docs, child_docs = load_and_split_parent_child(path)
        ok = vs.add_parent_child_documents(parent_docs, child_docs, get_file_info(path))

        # 父块 idx ↔ parent_id 映射（子块按 parent_id 回指）
        pid_to_idx = {p.metadata["parent_id"]: i for i, p in enumerate(parent_docs)}
        zw_count = sum(count_dirty_chars(c.page_content) for c in child_docs)

        out["documents"][did] = {
            "file": rel,
            "ok": ok,
            "elapsed_s": round(time.time() - t0, 1),
            "n_parents": len(parent_docs),
            "n_children": len(child_docs),
            "zero_width_ctrl_chars": zw_count,
            "parents": [
                {"i": i, "sha1": sha1(p.page_content), "len": len(p.page_content), "head": head(p.page_content)}
                for i, p in enumerate(parent_docs)
            ],
            "children": [
                {
                    "i": i,
                    "sha1": sha1(c.page_content),
                    "len": len(c.page_content),
                    "head": head(c.page_content),
                    "parent_idx": pid_to_idx.get(c.metadata.get("parent_id"), -1),
                }
                for i, c in enumerate(child_docs)
            ],
        }
        print(f"[parity-old]   {did}: parents={len(parent_docs)} children={len(child_docs)} ({out['documents'][did]['elapsed_s']}s)")

    hybrid = HybridRetriever(vs)
    vector_k = int(faiss_conf.get("vector_fetch_k", 20))
    bm25_k = int(faiss_conf.get("bm25_fetch_k", 20))

    print("[parity-old] running queries ...")
    for q in QUERIES:
        t0 = time.time()
        # ① 向量 top-k（带距离）
        vec = vs.faiss_store.similarity_search_with_score(q, k=vector_k)
        vector_items = [
            {"sha1": sha1(d.page_content), "dist": round(float(s), 6), "len": len(d.page_content)}
            for d, s in vec
        ]
        # ② BM25 top-k
        bm = hybrid._bm25_candidates(q, vs.tokenized_corpus, vs.child_documents, bm25_k)
        bm25_items = [{"sha1": sha1(d.page_content), "len": len(d.page_content)} for d in bm]
        # ③ 最终上下文（重排 + 动态 TopK + 父子块还原后的父块）
        context, provenance = hybrid.retrieve_for_rag(q)
        final_parents = []
        for p in provenance:
            blob = vs.parent_store.get(p["parent_id"], {})
            text = blob.get("page_content", "")
            final_parents.append({"sha1": sha1(text), "source": p.get("source", ""), "len": len(text)})

        out["queries"].append(
            {
                "q": q,
                "elapsed_s": round(time.time() - t0, 1),
                "vector_top": vector_items,
                "bm25_top": bm25_items,
                "final_parents": final_parents,
                "context_chars": len(context),
            }
        )
        print(f"[parity-old]   Q: {q}  final_parents={len(final_parents)} context_chars={len(context)} ({out['queries'][-1]['elapsed_s']}s)")

    result_path = os.path.join(REPO_ROOT, "eval_results", "parity", "old_stack.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[parity-old] saved -> {result_path}")


if __name__ == "__main__":
    main()
