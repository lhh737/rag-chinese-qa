"""把公开电商客服数据集（ecom-cs-agent）转成第二域 RAG 语料。

输入（本地，不提交）：
    data/business_faq/_raw/ecom_orders.json
    data/business_faq/_raw/ecom_logistics.json
    data/business_faq/_raw/ecom_customers.json
    data/business_faq/_raw/ecom_stock.json
    data/business_faq/_raw/ecom_policies.md

输出：
    20 个订单档案（order_A1001.txt ...）+ 10 篇政策文档（policy_01_*.txt ...）
    SOURCES.md / manifest.json

边界：
- 数据源为 GitHub 公开项目 guow7394/ecom-cs-agent，仓库无 LICENSE 文件；
  仅本地评测使用，公开仓库只提交题目/报告/来源链接，不转载全文。
- 订单/物流为模拟数据，报告里必须如实写“公开数据集（mock）”，不能写成真实京东订单。
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "business_faq"
RAW_DIR = CORPUS_DIR / "_raw"

SOURCE_REPO = "https://github.com/guow7394/ecom-cs-agent"

# 政策节合并：10 篇（覆盖退换货、退款、运费、地址/物流、保修、价格/发票、大额、人工）
POLICY_GROUPS = [
    ("policy_01_returns", "退换货总则（七天无理由与定制商品）", ["七天无理由退货", "定制商品"]),
    ("policy_02_quality", "质量问题退换货", ["质量问题退换货"]),
    ("policy_03_exchange", "换货规则", ["换货规则"]),
    ("policy_04_refund", "退款时效与到账", ["退款时效与到账"]),
    ("policy_05_shipping", "运费与运费险", ["运费与运费险"]),
    ("policy_06_address_logistics", "地址修改与物流异常", ["修改收货地址", "物流异常与催件"]),
    ("policy_07_warranty", "保修政策", ["保修政策"]),
    ("policy_08_price_invoice", "价格保护与发票", ["价格保护与补差价", "发票"]),
    ("policy_09_large_order", "大额订单售后", ["大额订单售后"]),
    ("policy_10_human", "人工客服与升级", ["人工客服"]),
]


def _load(name: str):
    return json.loads((RAW_DIR / name).read_text(encoding="utf-8"))


def _parse_policies(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def main() -> int:
    orders = _load("ecom_orders.json")
    logistics = _load("ecom_logistics.json")
    customers = _load("ecom_customers.json")
    stock = _load("ecom_stock.json")
    policies = _parse_policies((RAW_DIR / "ecom_policies.md").read_text(encoding="utf-8"))

    manifest: list[dict] = []

    # 1) 订单档案（20 篇）
    for order in orders:
        oid = order["order_id"]
        cust = customers.get(order.get("customer_id"), {})
        item = order.get("item", {})
        tracking = order.get("tracking_no")
        timeline = logistics.get(tracking, []) if tracking else []
        sku = item.get("sku")
        size = item.get("size")
        stock_line = ""
        if sku in stock:
            sizes = stock[sku]
            stock_line = "；".join(f"{k} 码 {v} 件" for k, v in sizes.items())
            if size and size in sizes:
                stock_line += f"（本单尺码 {size}：{sizes[size]} 件）"

        lines = [
            f"# 订单 {oid}",
            "",
            f"- 顾客：{cust.get('name', order.get('customer_id'))}（{order.get('customer_id')}，{cust.get('level', '普通会员')}）",
            f"- 商品：{item.get('name', '')}（SKU {sku}，{item.get('color', '')}，{item.get('size', '')}）"
            + ("，定制商品" if item.get("custom") else ""),
            f"- 金额：{order.get('amount')} 元",
            f"- 状态：{order.get('status')}",
            f"- 下单时间：{order.get('created_at', '—')}；发货时间：{order.get('shipped_at', '—')}；签收时间：{order.get('delivered_at', '—')}",
            f"- 收货地址：{order.get('address', '—')}",
            f"- 承运商：{order.get('carrier', '—')}；运单号：{tracking or '—'}",
        ]
        if stock_line:
            lines.append(f"- 库存：{stock_line}")
        if timeline:
            lines += ["", "## 物流轨迹", ""]
            lines += [f"- {t.get('time')} {t.get('desc')}" for t in timeline]
        else:
            lines += ["", "## 物流轨迹", "", "- 暂无物流记录"]
        content = "\n".join(lines).strip()
        path = CORPUS_DIR / f"order_{oid}.txt"
        path.write_text(content + "\n", encoding="utf-8")
        manifest.append({
            "doc_id": f"order_{oid}", "type": "order", "title": f"订单 {oid}",
            "source": "ecom-cs-agent/orders.json + logistics.json + customers.json + stock.json",
            "chars": len(content), "file": path.name,
        })

    # 2) 政策文档（10 篇）
    for doc_id, title, section_names in POLICY_GROUPS:
        parts = []
        for name in section_names:
            if name in policies:
                parts.append(f"## {name}\n\n{policies[name]}")
            else:
                print(f"  ! 政策节缺失：{name}")
        content = f"# 售后政策：{title}\n\n" + "\n\n".join(parts)
        path = CORPUS_DIR / f"{doc_id}.txt"
        path.write_text(content.strip() + "\n", encoding="utf-8")
        manifest.append({
            "doc_id": doc_id, "type": "policy", "title": f"售后政策：{title}",
            "source": "ecom-cs-agent/policies.md",
            "chars": len(content), "file": path.name,
        })

    # 3) SOURCES.md + manifest
    lines = [
        "# 电商客服 FAQ 语料来源（第二域）",
        "",
        f"> 生成日期：{date.today().isoformat()}",
        f"> 数据源：{SOURCE_REPO}（公开 GitHub，**无 LICENSE 文件**）",
        "> 用途：仅本地 mini eval；公开仓库只提交题目/报告/来源链接，不转载全文。",
        "> 性质：订单/物流/客户/库存为**模拟数据**，不是真实交易；报告必须如实写“公开数据集（mock）”。",
        "",
        "| doc_id | 类型 | 标题 | 来源文件 | 字符数 |",
        "|---|---|---|---|---:|",
    ]
    for row in manifest:
        lines.append(f"| {row['doc_id']} | {row['type']} | {row['title']} | {row['source']} | {row['chars']} |")
    (CORPUS_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (CORPUS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"✓ 订单档案 {sum(1 for x in manifest if x['type'] == 'order')} 篇")
    print(f"✓ 政策文档 {sum(1 for x in manifest if x['type'] == 'policy')} 篇")
    print(f"✓ 合计 {len(manifest)} 篇 → {CORPUS_DIR}")
    print(f"✓ 来源表 → {CORPUS_DIR / 'SOURCES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
