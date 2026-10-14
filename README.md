<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  中文 RAG 文档问答系统 · <b>以评测体系为主轴的质量工程</b><br>
  混合检索(RRF) · 重排序 · 父子块 · 行内引用 [n] · <b>自研 Judge（TPR/TNR 校准）</b> · 归因闭环
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-de5fe9" alt="uv">
  <img src="https://img.shields.io/badge/tests-68%20passed-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/cost-%E2%89%A4%C2%A531%20total-informational" alt="cost">
</p>
<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

> 这不是一个"又上线了一个 RAG Demo"，而是一次**围绕评测体系的 RAG 质量工程**：
> 六环闭环「构建 → 评测 → 可见 → 归因 → 迭代 → 安全」全部落地，每个数字都可回溯到
> Run Registry 与决策记录。核心主张：**先有尺子，再谈优化**——Judge 经 164 条人工盲标
> 校准（零分歧）后才用于主指标；工具（RAGAS）先验证再决定信任与否。

---

## 结果速览（全部可复现，见 `eval_results/`）

<p align="center"><img src="assets/results.svg" width="760" alt="关键结果速览：基线 vs Holdout 验收；消融结论"></p>

| 指标（质量集 dev 60 / holdout 20） | 基线（R1） | **Holdout 验收** |
|---|---|---|
| Recall@5（文档级） | 0.717 | **0.800** |
| 误拒率（质量题） | 0.231 | **0.167** |
| 答案正确性（judge） | 0.692 | **0.667** |
| 忠实度 / 引用准确性（judge） | 0.981 / 0.923 | **1.000 / 1.000** |
| 库外拒答率 | 1.000 | 1.000 |

- **Judge 校准**：164 条人工盲标 × 二分判定 → 与校准后的 judge **零分歧**；
  第二裁判交叉 34/34、验收集异族锚点 10/10（完整 TPR/TNR/κ 见
  [`P4 校准报告`](eval_results/calibration/reports/P4_calibration_report.md)）。
- **移除式消融（配对 McNemar，10 个对照运行）**：**重排是唯一显著的边际贡献**
  （去掉后误拒 +13.5pp，p=0.023）；HyDE 净贡献 ≈0（成本 −45% → 默认关闭，DR-003）；
  RRF k 扫描为负结果；并定量出检索指标的 **run 间噪声地板（±3–5pp）**。
- **归因**：规则引擎把 19 条失败归入 5 类（多跳单边召回 / 截断线 / 综合失败 / 遗漏 / 引用），
  人工抽检一致率 **20/20**；失败最大缺口（多跳第二来源召回）连同负结果如实留档。
- **工具对照**：RAGAS 与人工一致率 50–65%，三项中文失效模式实证（拒答语义/粒度过严/引用不可知）
  → 不进主链路；DeepEval 按设计退回 pytest 断言层。**"连评测工具都先验证"**。

---

## 系统架构

<p align="center"><img src="assets/architecture.svg" width="760" alt="系统架构：检索管线与评测体系"></p>

- **父子块**：子块（400 字）检索、父块（1200 字）还原上下文；
- **行内引用**：答案句 ↔ `[n]` ↔ 上下文块逐句可解析（含编号合法性代码校验）；
- **统一门面**：CLI / Gradio / FastAPI / 评测引擎共用 `RAGPipeline`，离线零 LangChain（DR-001）。

## 评测体系（六环闭环）

<p align="center"><img src="assets/eval-loop.svg" width="760" alt="评测主轴：六环闭环"></p>

| 环 | 交付物 |
|---|---|
| 构建 | `data/golden/`：质量集 80（配额分层 + dev/holdout）+ 安全集 20（含跨块拆分注入）；判定规范 `docs/annotation-guideline.md` |
| 评测 | `src/ragqa/evaluation/`：Run Registry（配置/版本/成本全回溯）、judged 执行器、自研 Judge（缓存 + 防漂移） |
| 可见 | `apps/review_viewer.html`：单文件查看器（审校/盲标/抽检/归因复核四模式） |
| 归因 | `ragqa.evaluation.attribution`：taxonomy v1 规则引擎（证据可查）+ 人工抽检闭环 |
| 迭代 | 移除式消融 / k 扫描 / prompt 与配置 A/B，配对检验裁定；决策记录 `decisions/DR-00x` |
| 安全 | 注入三层（query/doc/跨块）+ 有害 + 越界；规则 + judge 双层，防御前后对照 |

## 快速开始

> 前置：Python 3.12+、[uv](https://docs.astral.sh/uv/)；API Key：DashScope（向量/重排）+ DeepSeek（生成/裁判）。

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git && cd rag-chinese-qa
uv sync && cp .env.example .env    # 填入 DASHSCOPE_API_KEY / GEN_API_KEY / JUDGE_*

uv run ragqa ingest                # 构建索引（105 篇论文 + 中文文档，幂等可续跑）
uv run python apps/gradio_ui.py    # 问答界面 http://127.0.0.1:7860
uv run ragqa serve                 # FastAPI：/health /documents /query（http://127.0.0.1:8000/docs）
```

**评测与校准**：

```bash
uv run python scripts/run_eval.py --suite quality --split dev --mode judged   # judged 全量评测
uv run python scripts/calibrate_judge.py --segment tuning --labels <盲标导出>   # 三段式 judge 校准
uv run python scripts/attribute_run.py --run-id <run_id>                      # 自动归因 + 抽检文件
uv run pytest -q                                                              # 68 项测试（离线）
```

## 复现性与工程纪律

- **Run Registry**（`eval_results/registry.jsonl`）：每次运行的类型/配置/版本/指标/成本；
  trace 含完整配置指纹（`config_id`，如 `vec24_bm2524_rrf60_rr12_k4-8_hyde0_rr1bm1`）。
- **防漂移**：judge prompt / 模型变更 → 重跑测量段校准；缓存键含 prompt 内容哈希。
- **成本**：全项目 P0–P6 累计 ≈ **¥31**（含 10 个 judged 全量运行 + 校准 + 验收，含裁判与重排）。
- 评测集 / judge 版本冻结 + hash 记录（`data/golden/CHANGELOG.md`）。

## 目录速览

```
src/ragqa/            核心包（core 门面 / retrieval / generation / evaluation / api）
scripts/              运行与实验五件套：run_eval · calibrate_judge · attribute_run · judge_crosscheck · rejudge_errors
eval_results/         全部运行产物 + 分析报告（analysis/ 下：基线与 P6 报告、消融面板、k 扫描、RAGAS 对照）
decisions/            决策记录（DR-001 依赖迁移 / DR-002 事故 / DR-003 HyDE / DR-004 迭代裁定）
docs/                 评测方案 · 判定规范 · 标签表 · P5 计划书
data/golden/          冻结评测集（golden_v1.1 / safety_v1 + CHANGELOG + 健康度报告）
```

## 已知局限（如实声明）

- **单标注人**：标注一致性 26/30（隔日复标）；归因抽检 20/20——样本量小，区间宽；
- **裁判**：主裁判与生成同为 DeepSeek 族（单族预案，已声明自偏好局限）；异族锚点仅抽样；
- **噪声地板**：检索层指标 run 间波动 ±3–5pp，小于此的差值不作结论；
- **开缺口**：多跳题第二来源"从未召回"（候选池层问题）——k/TopK/HyDE/prompt 四类廉价手段均已实证无效，
  需切分 / embedding / 多查询检索层工作（未来方向）；
- 安全 judge 为轻校准（15 条人工对照，见 P6 报告）；`s005` 类"因注入放弃任务"软失败已记录。

## 后续方向

多跳检索层改造（第二查询 / 来源感知切分）→ 其配对复验；HyDE 约束化版本对照；
安全 judge 扩样校准；`ragqa serve` 生产化（鉴权 / 限流）。

<p align="center"><sub>MIT License · 评测方法论参考：Hamel Husain《Your AI Product Needs Evals》等</sub></p>
