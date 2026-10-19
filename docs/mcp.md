# MCP Server

`ragqa mcp` 把检索、取全文、文档列表和问答暴露为标准 MCP 工具，供任意 MCP 客户端调用。实现按 MCP 规范 **2026-07-28** / Python SDK v2（`MCPServer`），工具是 `RAGPipeline` 的薄封装，不另写检索逻辑。

## 安装与运行

```bash
uv sync --extra mcp
uv run ragqa mcp                                # stdio（由客户端拉起；stdout 仅走协议消息）
uv run ragqa mcp --transport http --port 8000   # Streamable HTTP → http://127.0.0.1:8000/mcp
```

> 运行时必须带 `--extra mcp`。裸跑 `uv sync` 会把 optional extra 删掉，MCP server 会启动失败。

## 工具

| 工具 | 作用 |
|---|---|
| `search_documents` | 混合检索，返回相关块与来源 |
| `get_document_content` | 按 doc_id 取原文 |
| `list_documents` | 列出索引内文档 |
| `answer` | 走完整 RAG 管线，返回带 `[n]` 引用的答案 |

## 客户端注册

stdio 方式：

```json
{"mcpServers": {"rag-chinese-qa": {"type": "stdio", "command": "uv",
  "args": ["run", "--directory", "<仓库路径>", "ragqa", "mcp"]}}}
```

或：

```bash
claude mcp add --scope project ragqa -- uv run --directory <仓库路径> ragqa mcp
```

依赖变更（放开 pydantic 上限以兼容 MCP SDK v2）记录在 `decisions/DR-005-mcp-deps.md`。
