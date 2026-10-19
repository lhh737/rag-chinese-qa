"""抓取公开电商帮助中心文档（第二域 mini eval 语料）。

用法：
    uv run python scripts/fetch_business_faq.py --limit 25
    uv run python scripts/fetch_business_faq.py --seeds URL1 URL2 --limit 10

支持两类京东页面：
- 桌面帮助页 `help.jd.com/user/issue/...`（正文在 `sl-paragraph`）；
- 移动帮助页 `in.m.jd.com/help/*.html`（正文在 `help-tit01` / `help-cnt` / `help-listq01`）。

边界：
- 只抓公开帮助页；限速 0.8s；只取一层链接；
- 保存标题/URL/抓取日期；正文落 `data/business_faq/`，是否公开提交由作者决定；
- 跳过隐私/权限/公司介绍/资质页，只保留购物、支付、配送、发票、售后、优惠等客服主题；
- **2026-10-19 实测**：京东移动站会 302 到 WAF error 页，桌面长政策页可抓；本脚本是 best-effort
  采集辅助，第二域最终语料由 `build_business_corpus.py` 从公开数据集生成。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "business_faq"

DEFAULT_SEEDS = [
    # 桌面站：两篇长政策（正文在 sl-paragraph）
    "https://help.jd.com/user/issue/133-988.html",
    "https://help.jd.com/user/issue/329-996.html",
] + [
    # 移动站：静态帮助页（索引页已会 302 到 error 页，这里直接列文章）
    f"https://in.m.jd.com/help/{name}"
    for name in [
        "register_info.html", "flow.html", "level.html", "score.html",
        "youhuijuan.html", "price-explanation.html", "jingdongpeisong.html",
        "shangjiapeisong.html", "address.html", "daofu.html", "zaixian.html",
        "ziti.html", "youju.html", "fapiao.html", "ziyingshouhou.html",
        "disanfangshouhou.html", "faq.html", "saomazhifu.html", "faqone.html",
        "tuihuanzhengce.html", "faqfour.html", "faqfive.html", "faqsix.html",
        "faqseven.html", "faqeight.html", "faqnine.html", "faqten.html",
    ]
]

DESKTOP_ARTICLE_RE = re.compile(r"(?:https?:)?//help\.jd\.com/user/issue/(\d+(?:-\d+)?)\.html")
MOBILE_LINK_RE = re.compile(r'href=["\'](\./[^"\']+\.html)["\']')
MOBILE_URL_RE = re.compile(r"https://in\.m\.jd\.com/help/([^/]+\.html)")

# 隐私/权限/公司介绍/资质页：与客服 FAQ 无关，跳过
MOBILE_BLOCK = {
    "address.html", "address_list_authority.html", "camera_authority.html",
    "location_authority.html", "voice_authority.html", "upload_authority.html",
    "order_sharing_info.html", "qiyewenhua.html", "intro.html",
    "platformQualification/platform-qualification-list.html",
    "publishingLicense/publishing-license-list.html",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


class _DesktopExtractor(HTMLParser):
    """京东桌面帮助页：标题 help-tit1 + 正文 sl-paragraph。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.in_title = False
        self.in_p = False
        self.title_parts: list[str] = []
        self.para_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1
        if tag == "div" and "help-tit1" in cls:
            self.in_title = True
        if tag == "p" and "sl-paragraph" in cls:
            self.in_p = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1
        if tag == "div" and self.in_title:
            self.in_title = False
        if tag == "p" and self.in_p:
            self.in_p = False
            self.para_parts.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_title:
            self.title_parts.append(data)
        elif self.in_p:
            self.para_parts.append(data)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(enc)
            if text.count("�") == 0 and len(re.findall(r"[一-鿿]", text)) > 20:
                return text.lstrip("﻿")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").lstrip("﻿")


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_desktop(raw: bytes) -> tuple[str, str]:
    text = _decode(raw)
    parser = _DesktopExtractor()
    parser.feed(text)
    title = _clean("".join(parser.title_parts))
    content = _clean("".join(parser.para_parts))
    if not title:
        m = re.search(r"<title>(.*?)</title>", text, re.S)
        title = _clean(m.group(1)) if m else ""
    return title, content


def _extract_mobile(raw: bytes) -> tuple[str, str]:
    text = _decode(raw)
    m = re.search(r"<title>(.*?)</title>", text, re.S)
    title = _clean(m.group(1)) if m else ""
    body = re.search(r"<body[^>]*>(.*?)</body>", text, re.S)
    if not body:
        return title, ""
    fragment = re.sub(r"<script.*?</script>|<style.*?</style>", "", body.group(1), flags=re.S)
    # 块级标签换行，其余标签去掉
    fragment = re.sub(r"</?(?:h[1-6]|p|li|ul|ol|div|br|tr|td|th|section|article)[^>]*>", "\n", fragment,
                      flags=re.I)
    content = _clean(re.sub(r"<[^>]+>", "", fragment))
    return title, content


def _get(url: str, timeout: int = 25) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
        print(f"  ✗ {url} (HTTP {r.status_code})")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {url} ({exc})")
    return None


def _doc_id(url: str) -> str:
    m = DESKTOP_ARTICLE_RE.search(url)
    if m:
        return f"jd_{m.group(1).replace('-', '_')}"
    m = MOBILE_URL_RE.search(url)
    if m:
        stem = Path(m.group(1)).stem
        return f"jd_m_{re.sub(r'[^a-zA-Z0-9_]+', '_', stem)}"
    return "jd_" + hashlib.sha1(url.encode()).hexdigest()[:10]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 种子页 → 候选文章
    candidates: list[str] = []
    for seed in args.seeds:
        # 移动站索引页不是文章，需要抓取它的相对链接
        is_mobile_index = seed.rstrip("/").endswith("/help/index.html")
        if (DESKTOP_ARTICLE_RE.search(seed) or MOBILE_URL_RE.search(seed)) and not is_mobile_index:
            if seed not in candidates:
                candidates.append(seed)
            continue
        print(f"种子 {seed}")
        raw = _get(seed)
        if not raw:
            continue
        text = _decode(raw)
        if "in.m.jd.com" in seed:
            for m in MOBILE_LINK_RE.finditer(text):
                rel = m.group(1).lstrip("./")
                if rel in MOBILE_BLOCK:
                    continue
                url = f"https://in.m.jd.com/help/{rel}"
                if url not in candidates:
                    candidates.append(url)
        else:
            for m in DESKTOP_ARTICLE_RE.finditer(text):
                url = f"https://help.jd.com/user/issue/{m.group(1)}.html"
                if url not in candidates:
                    candidates.append(url)
        time.sleep(args.delay)

    candidates = candidates[: args.max_candidates]
    print(f"候选文章 {len(candidates)} 篇；目标 {args.limit} 篇")

    # 2) 抓文章 → 提取 → 去重
    manifest: list[dict] = []
    seen_hashes: set[str] = set()
    for url in candidates:
        if len(manifest) >= args.limit:
            break
        raw = _get(url)
        if not raw:
            continue
        if MOBILE_URL_RE.search(url):
            title, content = _extract_mobile(raw)
        else:
            title, content = _extract_desktop(raw)
        zh = len(re.findall(r"[一-鿿]", content))
        if zh < 150:
            print(f"  - 跳过（正文过短 {zh} 字）：{title or url}")
            time.sleep(args.delay)
            continue
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            print(f"  - 跳过（重复）：{title}")
            time.sleep(args.delay)
            continue
        seen_hashes.add(digest)

        doc_id = _doc_id(url)
        path = out_dir / f"{doc_id}.txt"
        path.write_text(
            f"# {title}\nsource_url: {url}\nplatform: 京东\nfetched_at: {date.today().isoformat()}\n---\n{content}\n",
            encoding="utf-8",
        )
        manifest.append({
            "doc_id": doc_id,
            "title": title,
            "url": url,
            "chars": len(content),
            "zh_chars": zh,
            "sha256": digest[:16],
            "file": path.name,
        })
        print(f"  ✓ {doc_id}  {title}  ({zh} 中文字)")
        time.sleep(args.delay)

    # 3) SOURCES.md + manifest
    lines = [
        "# 电商客服 FAQ 语料来源",
        "",
        f"> 抓取日期：{date.today().isoformat()} · 平台：京东帮助中心（公开页面）",
        "> 仅用于本地评测；公开仓库是否提交全文由作者按版权决定。",
        "",
        "| doc_id | 标题 | URL | 中文字数 |",
        "|---|---|---|---:|",
    ]
    for row in manifest:
        lines.append(f"| {row['doc_id']} | {row['title']} | {row['url']} | {row['zh_chars']} |")
    (out_dir / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n完成：{len(manifest)} 篇 → {out_dir}")
    print(f"来源表：{out_dir / 'SOURCES.md'}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
