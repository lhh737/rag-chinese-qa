# DR-001 · 依赖迁移：去 langchain 全家桶（P0）

> 阶段：P0 · 日期：2026-10-05 · 状态：**采纳**（对照通过，回退开关未启用）
> 设计依据：`design/TECHNICAL_DESIGN.md` v5.0 §5.4（含"迁移不阻塞骨架"的回退条款）

## 问题

`langchain-community` 已 sunset（2026-06），仓库原有 3 处依赖它（FAISS 包装 / PyPDFLoader / TextLoader）；
且评测体系需要自有的结构化 trace、参数化分块与确定性的行号映射，中间层反而是不确定性来源。

## 方案

1. **`faiss-cpu` 直连**：`IndexFlatL2`（与旧 langchain FAISS 默认距离策略一致）；
   faiss 第 i 行 ↔ `child_documents[i]` 顺序落盘，加载期做 count / dim 双校验（不一致拒绝加载）。
2. **`pypdf` 直连**：逐页 `Document`（metadata: doc_id / source / page），替代 PyPDFLoader / TextLoader。
3. **自研 `RecursiveTextSplitter`**：按 langchain-text-splitters 1.1.x 的
   `_split_text / _split_text_with_regex / _merge_splits / _join_docs` 逐行移植（keep_separator / strip_whitespace 语义对齐）。
4. 自有 `Document` dataclass；嵌入客户端去 `Embeddings` 基类（鸭子类型接口）。
5. **移除**：langchain、langchain-community、langchain-core、langchain-text-splitters、requirements.txt；
   **引入** uv + pyproject.toml + uv.lock（Python 3.12；`faiss-cpu==1.13.2`、`pypdf==6.11.0` 钉在与对照环境相同版本）。

## 对照实验（通过标准 = 设计 §5.4"迁移前后同 5 条查询的检索结果一致"）

方法：迁移前用旧栈导出基线（全局环境，commit `5af6e15`，`scripts/parity_check_old.py`），迁移后同输入导出
（`scripts/parity_check_new.py`）。10 篇文档（8 论文 + 2 中文 txt）× 5 条查询；**hyde=off、融合=merge**
（把"依赖迁移"与"RRF 升级"两个变更分离，各自可归因）。
证据：`eval_results/parity/{old_stack,new_stack,compare_report,chunker_compare}.json`
（对照脚本为旧栈一次性产物，仅 `parity_check_new.py` 可重跑）。

| 检查项 | 结果 | 判定 |
|---|---|---|
| 分块器逐块比对（3 论文 + 2 txt，两组参数） | 417/417 父块、1141/1141 子块 **100% 一致** | ✓ |
| 全流程 chunk 边界（10 文档，5431 个位置） | **100.000%** | ✓ |
| BM25 top-24 排序（5 查询） | **5/5 完全一致**（修复并列排序不稳定性后） | ✓ |
| 最终上下文（父块）来源序列（5 查询） | **5/5 一致** | ✓ |
| 向量 top-24 | 集合重叠 0.92–1.00（均值 0.936），排序在近并列处扰动 | ✔ 见下"发现 2" |

**结论：迁移通过**；回退开关（保留 langchain-text-splitters）未启用。

## 过程中发现并处理的问题

1. **BM25 并列排序不稳定**（本次对照发现，已修复）：`np.argsort` 默认 quicksort 对并列分数顺序不确定，
   旧栈 `sorted(..., reverse=True)` 是稳定排序。已改为 `np.argsort(-scores, kind="stable")` 并加单测。
   修复前"什么是大语言模型？"的 top-24 仅 4/24 重合，修复后 5/5 完全一致。
2. **嵌入 API 非确定性（重要，评测阶段需纳入噪声模型）**：同一文本重复调用实测——
   - **query 路径**：3 次调用逐位相同（max|Δ| = 0，cos = 1.00000000）；
   - **document 路径**：2 次调用 max|Δ| = 7.5e-3（cos = 0.99809）。
   ⇒ 每次**索引重建**都会引入 ~1e-2 量级的距离噪声，近并列排序随之扰动——本次向量集合差异即此来源，
   与迁移无关（旧栈重复运行同样会出现）。**对评测的影响**：P5 稳定性抽检需量化 run-to-run 噪声带；
   发布门禁"回退超噪声带"的判定必须覆盖"索引重建"这一天然噪声源。
3. **旧代码 bug**：`model/dashscope_embedding.py` 引用了未导入的 `LLM_API_KEY`（NameError），对照前修复（`5af6e15`）。
4. **清洗影响审计**：10 篇对照文档中仅 `2303.08774` 含 23 个零宽/控制字符；对照时**关闭清洗**以隔离迁移变更，
   正式建索引开启清洗（该文档 chunk 边界会相应变化，属预期内、已记录）。

## 遗留与偏差

- CLI 实现于 `src/ragqa/cli.py`（console script `ragqa`），与设计 §8 所示 `apps/cli.py` 略有偏差
  （src 布局下作为包内模块才能打包为命令）。
- manifest 仍为 JSON（已含 status / content_hash / 耗时 / 清洗统计）；SQLite 迁移留 P7。
- 索引快照 / 回滚留 P7；当前重建 = 重跑 `ragqa ingest`。
