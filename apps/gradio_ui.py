from __future__ import annotations

import os
import shutil

import gradio as gr
import pandas as pd

from config.settings import PAPERS_DIR, UPLOAD_DIR
from rag.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None

CSS = """
.doc-list { font-size: 13px; }
.upload-area { margin-bottom: 12px; }
"""


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(use_hybrid=True, use_hyde=True)
    return _pipeline


def refresh_file_list() -> tuple[pd.DataFrame, gr.Dropdown]:
    p = get_pipeline()
    files = p.get_file_list()
    if not files:
        df = pd.DataFrame(columns=["文件名", "大小(KB)", "载入时间", "MD5"])
        return df, gr.Dropdown(choices=[], value=None, interactive=True)
    data = []
    choices = []
    for f in files:
        data.append([f["filename"], f["size_kb"], f["loaded_at"], f["md5"]])
        choices.append(f"{f['filename']} ({f['loaded_at']})")
    df = pd.DataFrame(data, columns=["文件名", "大小(KB)", "载入时间", "MD5"])
    return df, gr.Dropdown(choices=choices, value=choices[0], interactive=True)


def upload_and_index(file_obj) -> tuple[str, pd.DataFrame, gr.Dropdown]:
    if file_obj is None:
        return "请先选择文件", *refresh_file_list()

    src = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    filename = os.path.basename(src)
    dest = os.path.join(UPLOAD_DIR, filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    shutil.copy2(src, dest)

    p = get_pipeline()
    ok, msg = p.upload_and_index(dest)
    df, dd = refresh_file_list()
    return msg, df, dd


def batch_import_papers() -> tuple[str, pd.DataFrame, gr.Dropdown]:
    if not os.path.isdir(PAPERS_DIR):
        return f"论文目录不存在: {PAPERS_DIR}", *refresh_file_list()

    papers = []
    for dirpath, _, filenames in os.walk(PAPERS_DIR):
        for f in filenames:
            if f.endswith((".pdf", ".txt")) and not f.startswith("."):
                papers.append(os.path.join(dirpath, f))

    if not papers:
        return f"在 {PAPERS_DIR} 下未找到论文文件", *refresh_file_list()

    p = get_pipeline()
    ok_count = skip_count = fail_count = 0
    for fp in sorted(papers):
        ok, msg = p.upload_and_index(fp)
        if ok:
            ok_count += 1
        elif "已存在" in msg:
            skip_count += 1
        else:
            fail_count += 1

    df, dd = refresh_file_list()
    total = p.get_file_list()
    report = (
        f"批量导入完成：成功 {ok_count}，跳过 {skip_count}，失败 {fail_count}"
        f"（知识库共 {len(total)} 篇文档）"
    )
    return report, df, dd


def chat_fn(message: str, history: list):
    if not history or not isinstance(history, list):
        history = []
    if not message.strip():
        return history, ""
    p = get_pipeline()
    try:
        result = p.query_with_sources(message.strip())
        answer = result["answer"]
        sources = result.get("sources", [])
        if sources:
            src_lines = ["\n\n**参考来源**"]
            seen = set()
            for s in sources:
                src = s.get("source", "")
                if src and src not in seen:
                    seen.add(src)
                    excerpt = s.get("excerpt", "")[:120]
                    src_lines.append(f"- **{src}**: {excerpt}…")
            answer += "\n" + "\n".join(src_lines)
    except Exception as e:
        answer = f"生成回答时出错: {e}"

    history.append((message, answer))
    return history, ""


def view_document(selected: str) -> str:
    if not selected:
        return "请先选择一个文档"
    p = get_pipeline()
    files = p.get_file_list()
    for f in files:
        label = f"{f['filename']} ({f['loaded_at']})"
        if label == selected:
            content = p.get_document_content(f["md5"])
            if content:
                return f"```\n{content[:3000]}\n```" + (
                    "\n\n*...内容过长，仅显示前 3000 字符*" if len(content) > 3000 else ""
                )
            return f"无法读取文件内容: {f['filename']}"
    return "未找到该文档"


def build_ui():
    with gr.Blocks(title="RAG 中文文档问答系统") as demo:
        gr.Markdown(
            "## RAG 中文文档问答系统  \n"
            "技术栈：HyDE + 混合检索(BM25/向量) + Reranker + Qwen API  \n"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 文档管理")
                with gr.Group(elem_classes="upload-area"):
                    upload_file = gr.File(label="上传文档（txt/pdf）", file_types=[".txt", ".pdf"])
                    upload_btn = gr.Button("上传并解析", variant="primary")
                    upload_msg = gr.Textbox(label="状态", interactive=False)

                batch_btn = gr.Button("批量导入 papers/ 论文", variant="secondary")

                gr.Markdown("### 已载入文档")
                doc_list = gr.Dataframe(
                    headers=["文件名", "大小(KB)", "载入时间", "MD5"],
                    datatype=["str", "number", "str", "str"],
                    interactive=False,
                    elem_classes="doc-list",
                )
                doc_selector = gr.Dropdown(label="选择文档查看", choices=[], interactive=True)
                view_btn = gr.Button("查看文档内容")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("对话"):
                        chatbot = gr.Chatbot(height=420, label="多轮对话（含答案溯源）")
                        with gr.Row():
                            msg = gr.Textbox(
                                label="输入问题",
                                placeholder="输入问题后按回车发送...",
                                scale=4,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)

                    with gr.TabItem("文档查看"):
                        doc_content = gr.Markdown("请先在左侧选择文档并点击「查看文档内容」")

        upload_btn.click(
            upload_and_index,
            inputs=[upload_file],
            outputs=[upload_msg, doc_list, doc_selector],
        )
        batch_btn.click(
            batch_import_papers,
            inputs=[],
            outputs=[upload_msg, doc_list, doc_selector],
        )
        view_btn.click(
            view_document,
            inputs=[doc_selector],
            outputs=[doc_content],
        )
        send_btn.click(
            chat_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        )
        msg.submit(
            fn=chat_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        )

        demo.load(refresh_file_list, outputs=[doc_list, doc_selector])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False, css=CSS)
