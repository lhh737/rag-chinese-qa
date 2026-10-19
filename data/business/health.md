# 第二域 mini eval 健康度

- 生成日期：2026-10-19
- 题数：30；题型：{'factual': 12, 'multi_hop': 6, 'colloquial': 4, 'out_of_scope': 4, 'adversarial': 4}
- hash：`f1ae94bc770a784c…`
- 全部 `split=dev`（mini eval，无 holdout；不作强统计结论）。
- 作者标注：`human_reviewed=false`（待人工审校后再改为 true）。

## 校验结果

通过。

## 防泄漏

无题干 >0.7 的 5-gram 泄漏。

## 口径提醒

- 语料为公开数据集 **ecom-cs-agent**（mock 订单/物流 + 政策），不是真实京东订单；报告必须如实写。
- 语料仅 30 篇，R@5 与论文域 106 篇不可直接比；重点看答案正确性、误拒、失败模式。
- 时间敏感题已在题干写明“以 2026-10-19 为今天”，避免依赖运行日期。
