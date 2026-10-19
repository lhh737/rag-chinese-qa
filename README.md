<h1 align="center">RAG-Chinese-QA</h1>
<p align="center">
  中文文档问答 · 评测工程<br>
  混合检索 / 重排 / 行内引用 / 校准 judge / 可回溯运行
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python">
  <img src="https://github.com/lhh737/rag-chinese-qa/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/uv-managed-de5fe9" alt="uv">
</p>
<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

## 项目简介

在 106 篇中文论文/文档（32,501 个子块）上构建 RAG 问答：FAISS + BM25 双路召回、RRF 融合、重排序；答案带 `[n]` 行内引用，库外问题明确拒答。

主要工作在评测层：80 道质量题 + 20 道安全题、164 条人工盲标校准的 judge、配对消融、失败归因、第二域迁移，以及覆盖每次查询的 trace 和 Run Registry。

值得参考的部分：评测集与 judge 校准、配对消融与负结果、第二域迁移、标注 / QC / 成本。

| | |
|---|---|
| 语料 | 106 篇 / 32,501 子块；第二域 30 篇 |
| 评测集 | 质量 80 + 安全 20；第二域 30 |
| 运行档案 | 28 次 run（`eval_results/registry.jsonl`） |
| 测试 | 90 项离线测试 + GitHub Actions |
| 模型 | qwen3.7-text-embedding / rerank；deepseek-flash 生成与裁判 |
| 成本 | P0–P6 ≈¥31；第二域 run ¥0.20 |

> 个人项目，不是生产服务；重点是可复现的评测记录。

## 演示

<p align="center">
  <img src="assets/webui-qa.png" width="48%" alt="问答界面">
  <img src="assets/webui-refusal.png" width="48%" alt="库外问题拒答">
</p>
<p align="center"><sub>左：回答带 [n] 行内引用与检索溯源；右：库外问题明确拒答</sub></p>

## 结果

<p align="center"><img src="assets/results.svg" width="760" alt="R1 基线 → holdout 验收"></p>

| 指标 | 论文 dev 60（R1） | 论文 dev 60（最终） | 论文 holdout 20 | 电商 FAQ 30 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.717 | 0.758 | 0.800 | 0.825* |
| 误拒率 | 23.1% | 23.1% | 16.7% | 23.1% |
| 答案正确性 | 0.692 | 0.725 | 0.667 | 0.692 |
| 忠实度 / 引用 | 0.981 / 0.923 | 1.000 / 1.000 | 1.000 / 1.000 | 0.923 / 0.923 |
| 库外拒答 | 100% | 100% | 100% | 100% |

R1 = HyDE on；最终 = R1 + HyDE off。*第二域只有 30 篇语料，R@5 不能与 106 篇论文域直接比较，保留它只为展示同一套指标。

### 被驳回的改动

| 实验 | 结果 | 处理 |
|---|---|---|
| 移除重排 | 误拒 +13.5pp（p=0.023），正确性 −12.3pp | 保留重排 |
| 开启 HyDE | 质量无改善，成本 +44% | 关闭（DR-003） |
| RRF k 扫描 k=10/30/60/100 | 无显著差异 | 不作结论 |
| TopK 6–10 | 误拒 p=0.131，多跳无改善 | 驳回 |
| Prompt v2 | 显著劣化（p=0.045） | 驳回 |
| Agentic 检索（单轮工具选择） | R@5 +1.7pp（噪声内）、MRR −13.1pp、多跳 8→10 | 驳回（P9） |

## 评测设计

<p align="center"><img src="assets/eval-loop.svg" width="760" alt="评测流程"></p>

| 环节 | 产物 |
|---|---|
| 构建 | 80 + 20 题；5 类配额；dev 60 / holdout 20；hash + CHANGELOG |
| 运行 | judged runner；trace / registry；逐题成本与延迟 |
| 查看 | `apps/review_viewer.html`：读 trace / 盲标 / 抽检 / 候选审校 |
| 归因 | 规则引擎 + 人工抽检（20/20） |
| 迭代 | 配对消融、McNemar、决策记录 |
| 安全 | 注入三层 + 有害 + 越界；防御前后对照 |

<p align="center"><img src="assets/reviewer-trace.png" width="760" alt="评审查看器：读 trace"></p>
<p align="center"><sub>读 trace：召回与融合明细、重排分数、最终上下文、生成与引用、成本总账</sub></p>

## 第二域

| | 论文域 | 电商客服 FAQ |
|---|---|---|
| 语料 | 106 篇 / 32,501 块 | 30 篇（订单 + 政策） |
| 题数 | dev 60 + holdout 20 | 30（全 dev，mini eval） |
| 失败主因 | 检索 8/19 | 日期算术、订单指代、政策分支、安全但无用 8/26 |

同一套最终配置和 judge 在第二域上：R@5 0.825、误拒 23.1%、答案正确性 0.692、忠实/引用 0.923、库外拒答 100%。失败面从“检索不到”转到“生成侧推理”，说明题型配额需要按域重设。

详见 [`docs/transferability.md`](docs/transferability.md)。

## Judge 校准与一致性

| 检查 | 结果 |
|---|---|
| 164 条人工盲标 | 164/164 有效；TPR/TNR 1.000 |
| 第二裁判（20%） | 34/34 |
| 异族锚点 | 10/10 |
| 第二域迁移（20 题 / 56 条） | agreement 0.946、κ 0.825、TPR 1.000、TNR 0.936 |
| AI 第二标注（同族，50 题 / 134 条） | agreement 0.948、κ 0.836；2 条 AI 过严、4 条规范缺口、1 条人工漏判 |

AI 第二标注与 judge/生成同族，报告的是 AI–human 一致性，不是人类标注者间 κ；它可以做预筛，不能替代人工抽检。

## 系统架构

<p align="center"><img src="assets/architecture.svg" width="760" alt="系统架构"></p>

- 检索：FAISS + BM25 → RRF → 重排；父子块（子块 400 字检索 / 父块 1200 字还原）。
- 上下文：动态 TopK 4–8；`[n]` 编号硬校验；拒答由 prompt + 零命中回退共同判定。
- 门面：CLI / Web UI / FastAPI / MCP / 评测共用 `RAGPipeline`；离线零 LangChain（DR-001）。
- 可观测：每次查询一条 trace，含各阶段延迟与成本；run 记录配置指纹。

### 技术栈

| 层 | 技术 |
|---|---|
| 检索 | FAISS、BM25、RRF、qwen3.7-text-rerank |
| 生成 | DeepSeek deepseek-flash |
| 服务 | FastAPI（SSE）、单文件 Web UI、MCP 2026-07-28 |
| 评测 | 自建 judge、McNemar / Wilson、JSONL trace、Run Registry |
| 运营 | SQLite QC 看板、标注 SOP、badcase 回归 |

## 运营与成本

- 标注/QC：[`docs/ops-playbook.md`](docs/ops-playbook.md)（7 维二分、20% 分层抽检、返修升级、badcase 回流）。
- 成本–质量：[`docs/cost-quality.md`](docs/cost-quality.md)（同集合 dev 60；HyDE off 质量不变、成本 −44%、p50 −45%；移除重排省 84% 但误拒 +13.5pp）。
- QC 看板：[`eval_results/analysis/qc_dashboard.md`](eval_results/analysis/qc_dashboard.md)（6 条 SQL，SQLite，零新增依赖）。

## 快速开始

> 前置：Python 3.12+、[uv](https://docs.astral.sh/uv/)；DashScope（向量 / 重排）+ DeepSeek（生成 / 裁判）API Key。

```bash
git clone https://github.com/lhh737/rag-chinese-qa.git && cd rag-chinese-qa
uv sync && cp .env.example .env    # 填入 DASHSCOPE_API_KEY / GEN_API_KEY / JUDGE_*

uv run ragqa ingest                # 构建索引（106 篇，幂等可续跑）
uv run ragqa serve                 # Web UI + FastAPI → http://127.0.0.1:8000
uv run pytest -q                   # 90 项离线测试
```

评测与校准：

```bash
uv run python scripts/run_eval.py --suite quality --split dev --mode judged
uv run python scripts/calibrate_judge.py --segment tuning --labels <盲标导出>
uv run python scripts/attribute_run.py --run-id <run_id>
```

MCP server 的安装、工具与客户端注册见 [`docs/mcp.md`](docs/mcp.md)。

## 项目结构

```
src/ragqa/            核心包：检索 / 生成 / 评测 / 服务 / MCP / 工具层
apps/                 Web UI + 评审查看器（单文件）
config/               检索 / 生成 / 计价配置
prompts/              answer / judge / annotator prompts
scripts/              评测、校准、归因、QC、回流、第二域脚本
eval_results/         运行产物 + analysis/ 报告 + regression/ 回归标签
data/golden/          冻结评测集（golden_v1.1 / safety_v1）
data/business/        第二域 30 题 + 健康度 + judge 抽检
data/business_faq/    第二域语料（本地；SOURCES.md 公开，全文不提交）
docs/                 方案、规范、运营、成本、迁移、MCP
decisions/            DR-001 ~ DR-005
```

## 文档

| 文档 | 回答的问题 |
|---|---|
| [docs/transferability.md](docs/transferability.md) | 同一套评测方法换到业务域还能用吗？ |
| [docs/ops-playbook.md](docs/ops-playbook.md) | 标注、QC、返修、badcase 回流怎么做？ |
| [docs/cost-quality.md](docs/cost-quality.md) | 每个质量点对应多少成本 / 延迟？ |
| [docs/annotation-guideline.md](docs/annotation-guideline.md) | 7 维标签怎么判？ |
| [docs/mcp.md](docs/mcp.md) | MCP server 怎么跑、怎么注册？ |
| [docs/index.md](docs/index.md) | 全部文档索引 |
| [eval_results/analysis/](eval_results/analysis/) | 基线、P6/P9、消融、k 扫描、QC 看板 |
| [decisions/](decisions/) | 关键设计选择与负结果 |

## 已知局限

- 单标注人；隔日复标 26/30；164/164 零分歧是 in-sample，样本量小。
- judge 与生成同族；异族锚点只有 10 条；第二域 judge 抽检 20 题 / 56 条。
- 检索指标 run 间噪声 ±3–5pp；小于此的差值不作结论。
- 第二域是 30 题 mini eval（全 dev、mock 数据）；b020/b021 题干缺 as-of 日期，剔除后 ac 18/24=0.750、误拒 4/24=16.7%。
- 多跳第二来源“从未召回”仍未解决；k / TopK / HyDE / prompt / 多查询均无效；剩余方向是切分策略 / embedding。

## 路线图

- 多跳缺口：切分策略 / embedding 模型，配对复验。
- 安全 judge 扩样校准；HyDE 约束化版本对照。
- `ragqa serve` 生产化（鉴权 / 限流）。
- 第二域 v1.1：补 as-of 日期后重跑；AI 第二标注用 v1.1 prompt 复跑。

<p align="center"><sub>MIT License</sub></p>
