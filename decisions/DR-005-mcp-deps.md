# DR-005 · 依赖变更：放开 pydantic 上限，引入 MCP SDK（P9）

> 阶段：P9 · 日期：2026-10-16 · 状态：**采纳**
> 关联：DR-001（依赖迁移 / 对照环境钉版本）、`pyproject.toml`、`tests/test_deps_guard.py`、`src/ragqa/mcp_server.py`

## 问题

为把已有的 RAG 能力暴露为标准 MCP 工具（对外可被任意 MCP 客户端调用），需要 mcp SDK。
但 `mcp>=2.0` 要求 `pydantic>=2.12.0`，而本仓库钉的是 `pydantic>=2.10,<2.11` —— 解析直接失败。

该上限**并非评测口径所需**，而是已移除的 gradio 的遗留约束（`pyproject.toml` 注释原文：「原为 gradio 兼容约束，
gradio 已移除，待评估放开」）。`<2.11` 在此已无技术依据，只剩"锁文件当时如此"的历史惯性。

## 方案

1. **放开上限**：`pydantic>=2.10,<2.11` → `pydantic>=2.12`。
2. **新增 optional extra**：`[project.optional-dependencies] mcp = ["mcp>=2.0"]` —— 对齐既有 `local` extra 的
   先例；核心链路不依赖它，仅 `ragqa mcp` 需要。
3. **延迟导入**：`mcp.server.MCPServer` 只在 `create_mcp_server()` 内导入，未装 extra 时其余功能不受影响。

## 验证（全部实测）

| 检查项 | 结果 | 判定 |
|---|---|---|
| 依赖解析 | pydantic `2.10.6` → **`2.14.1`**；新增 `mcp==2.3.0` + `mcp-types==2.3.0` 及 12 个传递依赖（cryptography / pyjwt / sse-starlette / jsonschema / attrs 等） | ✓ |
| **DR-001 对照钉版本未受影响** | `faiss-cpu==1.13.2`、`pypdf==6.11.0` 保持钉版不变 | ✓ 关键 |
| 回归测试 | 确认 `.venv` 内 pydantic 实为 2.14.1 后，全量 **71 项离线测试通过** | ✓ |
| 依赖边界 | 新增 `test_core_chain_does_not_load_mcp`：导入 core / cli / api / tools 后 mcp 模块**零新增** | ✓ |
| 端到端（真实 stdio 子进程 + 真实 MCP 客户端） | 协商协议版本 **`2026-07-28`**（非 legacy 回退）；4 工具可列可调；`list_documents` 返回 **106** 篇（与索引一致）；`search_documents("什么是大语言模型？")` → rank1 `llm_intro.txt` 0.970 / rank2 `2309.11495.pdf` 0.834 | ✓ |
| stdout 洁净（stdio 传输硬约束） | 真机实测：日志（`[mcp]` / jieba / httpx）全部落 stderr，协议在 stdout 正常通信 | ✓ |

**注**：先前一次"测试通过"的结论作废——那次 `.venv` 仍是 pydantic 2.10.6（`uv sync` 在下载阶段被超时截断），
跑的并非新环境。本表结论以「确认 venv 内版本后重跑」为准。

## 影响与回退

- **影响面**：pydantic 跨 4 个次版本，覆盖 config 模型（`config/loader.py`）、settings、API 模型（`api.py`）、
  评测数据结构。实测 71/71 通过，无需改代码。
- **CI**：`uv sync --frozen` → `uv sync --frozen --extra mcp`；否则 `tests/test_mcp_server.py` 会被
  `importorskip` **静默跳过**（与"每个数字可核验"的项目纪律冲突）。
- **回退**：移除 `mcp` extra 与 `src/ragqa/mcp_server.py` 即可；若同时收紧 pydantic 上限，须先移除 extra
  （否则重新出现不可满足的解析）。
- **未做**：远程 server 的鉴权/OAuth（`MCPServer` 已支持 `auth` / `token_verifier`，本轮未接）；未接入
  Langfuse / OpenTelemetry —— 可观测性仍以自研 Trace Schema + Run Registry 为准。
