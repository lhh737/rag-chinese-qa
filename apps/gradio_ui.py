"""Gradio 单入口 UI（瘦身版，设计 §7）：全部调用 RAGPipeline 门面，与评测对象同源。

运行：uv run python apps/gradio_ui.py  →  http://127.0.0.1:7860
"""
from __future__ import annotations

import os
import shutil

import gradio as gr
import pandas as pd

from ragqa.config.settings import get_settings
from ragqa.core import RAGPipeline
from ragqa.ingestion.pipeline import ingest_paths

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def refresh_file_list() -> tuple[pd.DataFrame, gr.Dropdown]:
    files = get_pipeline().get_file_list()
    if not files:
        return pd.DataFrame(columns=["文档", "子块", "KB"]), gr.Dropdown(choices=[], value=None)
    rows = [[f.get("doc_id"), f.get("n_children"), f.get("size_kb")] for f in files]
    df = pd.DataFrame(rows, columns=["文档", "子块", "KB"])
    choices = [f.get("doc_id", "") for f in files]
    return df, gr.Dropdown(choices=choices, value=choices[0] if choices else None, interactive=True)


def upload_and_index(file_obj):
    if file_obj is None:
        return "请先选择文件", *refresh_file_list()
    src = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    filename = os.path.basename(src)
    dest_dir = get_settings().upload_dir_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = str(dest_dir / filename)
    shutil.copy2(src, dest)

    results = ingest_paths([dest], store=get_pipeline().store)
    r = results[0]
    msg = f"{r.status}：{r.doc_id}（父 {r.n_parents} / 子 {r.n_children}）" if r.status == "indexed" else f"{r.status}：{r.doc_id} {r.error or ''}"
    return msg, *refresh_file_list()


def batch_import_papers():
    settings = get_settings()
    papers_dir = settings.papers_dir_path
    if not papers_dir.is_dir():
        return f"论文目录不存在: {papers_dir}", *refresh_file_list()

    results = ingest_paths([papers_dir, "data/llm_intro.txt", "data/test_transformer.txt"], store=get_pipeline().store)
    ok = sum(1 for r in results if r.status == "indexed")
    skip = sum(1 for r in results if r.status == "skipped")
    fail = sum(1 for r in results if r.status == "failed")
    total = len(get_pipeline().get_file_list())
    return f"批量导入完成：成功 {ok}，跳过 {skip}，失败 {fail}（知识库共 {total} 篇）", *refresh_file_list()


def chat_fn(message: str, history: list):
    history = list(history or [])
    if not message.strip():
        return history, ""
    try:
        res = get_pipeline().query(message.strip())
        answer = res.answer
        if res.contexts:
            lines = ["", "---", "**📎 参考来源**"]
            seen: set[str] = set()
            for b in res.contexts:
                if b.source and b.source not in seen:
                    seen.add(b.source)
                    lines.append(f"- `[{b.rank}]` {b.source}")
            answer += "\n".join(lines)
    except Exception as e:  # noqa: BLE001 —— UI 层兜底展示
        answer = f"生成回答时出错: {e}"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def view_document(selected: str) -> str:
    if not selected:
        return "请先选择一个文档"
    content = get_pipeline().get_document_content(selected)
    if not content:
        return f"无法读取文件内容: {selected}"
    shown = content[:3000]
    return f"```\n{shown}\n```" + ("\n\n*...内容过长，仅显示前 3000 字符*" if len(content) > 3000 else "")


CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; background: #f6f7f9 !important; }
footer { display: none !important; }
#hero {
  background: linear-gradient(135deg, #eef4fc 0%, #fcfcfb 65%);
  border: 1px solid #e5e4e0; border-radius: 14px; padding: 18px 24px 14px; margin-bottom: 6px;
}
#hero h1 { margin: 0 0 6px 0; font-size: 26px; letter-spacing: .5px; }
#hero .sub { color: #52514e; font-size: 13.5px; }
#hero .chips { margin-top: 10px; }
#hero .chips span {
  display: inline-block; background: #ffffff; border: 1px solid #dbe6f4; border-radius: 999px;
  padding: 2px 12px; margin: 3px 8px 3px 0; font-size: 12px; color: #2a78d6;
}
/* 卡片与控件细节 */
.gr-group, .block { box-shadow: 0 1px 3px rgba(15, 23, 42, .05) !important; }
button { border-radius: 10px !important; }
#status-box textarea { height: 38px !important; min-height: 38px !important; }
#status-box { min-height: 0 !important; }
table th { white-space: nowrap !important; }
"""

HERO = """
<div id="hero">
  <h1>RAG 中文文档问答</h1>
  <div class="sub">面向中文文档的检索增强问答 —— 以评测体系为主轴的质量工程（先有尺子，再谈优化）</div>
  <div class="chips">
    <span>混合检索 RRF</span><span>重排序</span><span>父子块</span><span>行内引用 [n]</span>
    <span>自研 Judge（盲标校准）</span><span>全程可回溯</span>
  </div>
</div>
"""

EXAMPLES = [
    ["Transformer 是哪个论文提出的？它提出了什么核心架构？"],
    ["RAG 是什么？它解决了什么问题？"],
    ["在 TriviaQA 的评测中，论文报告了哪两个测试集上的结果？"],
    ["请问在做冠脉搭桥手术时，通常选择哪条血管作为桥血管效果最好？"],
]


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=["-apple-system", "Segoe UI", "PingFang SC", "Microsoft YaHei", "sans-serif"],
    )
    with gr.Blocks(title="RAG 中文文档问答系统", theme=theme, css=CSS) as demo:
        gr.HTML(HERO)
        with gr.Row():
            with gr.Column(scale=1, min_width=330):
                with gr.Accordion("📚 文档管理（上传 / 批量导入）", open=False):
                    upload_file = gr.File(label="上传文档（txt/pdf）", file_types=[".txt", ".pdf"], height=120)
                    with gr.Row():
                        upload_btn = gr.Button("上传并解析", variant="primary", scale=1)
                        batch_btn = gr.Button("批量导入论文库", variant="secondary", scale=1)
                    status = gr.Textbox(label="状态", interactive=False, lines=1, max_lines=1, elem_id="status-box")
                with gr.Group():
                    gr.Markdown("#### 🗂 已索引文档")
                    doc_list = gr.Dataframe(headers=["文档", "子块", "KB"],
                                            interactive=False, max_height=220)
                    with gr.Row():
                        doc_selector = gr.Dropdown(label="查看内容", choices=[], interactive=True, scale=3)
                        view_btn = gr.Button("查看", variant="secondary", scale=1)
                    doc_content = gr.Markdown("")
            with gr.Column(scale=2, min_width=520):
                chatbot = gr.Chatbot(
                    height=520, label="对话（回答带 [n] 行内引用）", type="messages",
                    show_copy_button=True, placeholder="在下方输入问题开始…",
                    value=[{"role": "assistant", "content":
                            "👋 你好！我已加载本地文档库（左侧可管理文档）。直接提问即可 —— "
                            "回答会附带 `[n]` 行内引用与参考来源；库外问题会明确拒答。"}],
                )
                with gr.Row():
                    msg = gr.Textbox(placeholder="输入问题，回车发送…", show_label=False, scale=5, autofocus=True)
                    send_btn = gr.Button("发送 ➤", variant="primary", scale=1)
                gr.Examples(examples=EXAMPLES, inputs=msg, label="💡 试试这些问题")

        upload_btn.click(upload_and_index, inputs=[upload_file], outputs=[status, doc_list, doc_selector])
        batch_btn.click(batch_import_papers, inputs=[], outputs=[status, doc_list, doc_selector])
        view_btn.click(view_document, inputs=[doc_selector], outputs=[doc_content])
        send_btn.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        demo.load(refresh_file_list, outputs=[doc_list, doc_selector])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False, show_api=False)
