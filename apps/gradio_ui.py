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
        return pd.DataFrame(columns=["文档", "子块", "大小KB", "载入时间"]), gr.Dropdown(choices=[], value=None)
    rows = [[f.get("doc_id"), f.get("n_children"), f.get("size_kb"), f.get("loaded_at")] for f in files]
    df = pd.DataFrame(rows, columns=["文档", "子块", "大小KB", "载入时间"])
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
    history = history or []
    if not message.strip():
        return history, ""
    try:
        res = get_pipeline().query(message.strip())
        answer = res.answer
        if res.contexts:
            lines = ["\n\n**参考来源**"]
            seen: set[str] = set()
            for b in res.contexts:
                if b.source and b.source not in seen:
                    seen.add(b.source)
                    lines.append(f"- [{b.rank}] **{b.source}**")
            answer += "\n".join(lines)
    except Exception as e:  # noqa: BLE001 —— UI 层兜底展示
        answer = f"生成回答时出错: {e}"
    history.append((message, answer))
    return history, ""


def view_document(selected: str) -> str:
    if not selected:
        return "请先选择一个文档"
    content = get_pipeline().get_document_content(selected)
    if not content:
        return f"无法读取文件内容: {selected}"
    shown = content[:3000]
    return f"```\n{shown}\n```" + ("\n\n*...内容过长，仅显示前 3000 字符*" if len(content) > 3000 else "")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG 中文文档问答系统") as demo:
        gr.Markdown(
            "## RAG 中文文档问答系统\n"
            "HyDE + 混合检索（向量/BM25 + RRF）+ 重排 + 父子块 + 行内引用 [n]"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 文档管理")
                upload_file = gr.File(label="上传文档（txt/pdf）", file_types=[".txt", ".pdf"])
                upload_btn = gr.Button("上传并解析", variant="primary")
                batch_btn = gr.Button("批量导入 data/papers 论文", variant="secondary")
                status = gr.Textbox(label="状态", interactive=False)
                gr.Markdown("### 已索引文档")
                doc_list = gr.Dataframe(headers=["文档", "子块", "大小KB", "载入时间"], interactive=False)
                doc_selector = gr.Dropdown(label="选择文档查看", choices=[], interactive=True)
                view_btn = gr.Button("查看文档内容")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=460, label="对话（含 [n] 引用）")
                with gr.Row():
                    msg = gr.Textbox(label="输入问题", placeholder="回车发送…", scale=4)
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                doc_content = gr.Markdown("")

        upload_btn.click(upload_and_index, inputs=[upload_file], outputs=[status, doc_list, doc_selector])
        batch_btn.click(batch_import_papers, inputs=[], outputs=[status, doc_list, doc_selector])
        view_btn.click(view_document, inputs=[doc_selector], outputs=[doc_content])
        send_btn.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        demo.load(refresh_file_list, outputs=[doc_list, doc_selector])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)
