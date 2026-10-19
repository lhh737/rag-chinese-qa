# 第二域语料（本地生成物）

本目录的 `*.txt` 是 `scripts/build_business_corpus.py` 的生成物，按 `.gitignore` **不提交**；公开仓库只保留本 README、`SOURCES.md` 和题目/报告。

## 数据源

- `https://github.com/guow7394/ecom-cs-agent`
- **无 LICENSE 文件**：仅本地评测使用，不转载全文；引用时注明来源。
- 订单/物流/客户/库存为 **mock 数据**，不是真实交易。

## 复现

```bash
# 1) 下载原始文件到 _raw/
mkdir -p data/business_faq/_raw
base=https://raw.githubusercontent.com/guow7394/ecom-cs-agent/main/cs_agent/data
curl -L "$base/orders.json"    -o data/business_faq/_raw/ecom_orders.json
curl -L "$base/logistics.json" -o data/business_faq/_raw/ecom_logistics.json
curl -L "$base/customers.json" -o data/business_faq/_raw/ecom_customers.json
curl -L "$base/stock.json"     -o data/business_faq/_raw/ecom_stock.json
curl -L "$base/policies.md"    -o data/business_faq/_raw/ecom_policies.md

# 2) 生成 30 篇语料（20 订单档案 + 10 政策）
uv run python scripts/build_business_corpus.py

# 3) 建独立索引（不污染 faiss_db）
uv run ragqa ingest data/business_faq/order_*.txt data/business_faq/policy_*.txt --store faiss_db_business
```

> `scripts/fetch_business_faq.py` 是早期帮助中心抓取的 best-effort 辅助：京东移动站会 302 到 WAF error 页，桌面长政策页可抓；最终第二域语料以上面的公开数据集为准。
