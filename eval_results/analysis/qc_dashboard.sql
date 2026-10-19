-- QC 看板查询（SQLite 方言）
-- 表由 scripts/qc_metrics.py 从现有 JSONL 构建；本文件也可在 sqlite3 中复用（先建表/导入）。
-- 运行：uv run python scripts/qc_metrics.py
-- 口径：只读现有产物，不新跑实验；缺失字段显示为 NULL，不补造数字。

-- @title: 1. 标注合格率（按 criterion）
SELECT criterion,
       COUNT(*) AS n,
       SUM(CASE WHEN label = 'pass' THEN 1 ELSE 0 END) AS pass_n,
       SUM(CASE WHEN label = 'fail' THEN 1 ELSE 0 END) AS fail_n,
       ROUND(1.0 * SUM(CASE WHEN label = 'pass' THEN 1 ELSE 0 END) / COUNT(*), 4) AS pass_rate
FROM labels
GROUP BY criterion
ORDER BY criterion;

-- @title: 2. 分段合格率（segment × criterion）
SELECT segment, criterion,
       COUNT(*) AS n,
       SUM(CASE WHEN label = 'pass' THEN 1 ELSE 0 END) AS pass_n,
       ROUND(1.0 * SUM(CASE WHEN label = 'pass' THEN 1 ELSE 0 END) / COUNT(*), 4) AS pass_rate
FROM labels
GROUP BY segment, criterion
ORDER BY segment, criterion;

-- @title: 3. 单人复标一致性（10 题 × 3 criterion）
SELECT COUNT(*) AS compared,
       SUM(CASE WHEN l.label = r.label THEN 1 ELSE 0 END) AS same_n,
       SUM(CASE WHEN l.label <> r.label THEN 1 ELSE 0 END) AS changed_n,
       ROUND(1.0 * SUM(CASE WHEN l.label = r.label THEN 1 ELSE 0 END) / COUNT(*), 4) AS raw_agreement
FROM labels l
JOIN relabel r ON l.id = r.id AND l.criterion = r.criterion;

-- @title: 4. 数据集审校结果（按轮次）
SELECT round,
       SUM(CASE WHEN decision = 'pass' THEN 1 ELSE 0 END) AS pass_n,
       SUM(CASE WHEN decision = 'reject' THEN 1 ELSE 0 END) AS reject_n,
       SUM(CASE WHEN decision = 'undecided' THEN 1 ELSE 0 END) AS undecided_n,
       COUNT(*) AS total_n
FROM review
GROUP BY round
ORDER BY round;

-- @title: 5. 失败归因分布（排除“无问题”）
SELECT primary_class AS failure_class, COUNT(*) AS n
FROM attribution
WHERE primary_class <> '无问题'
GROUP BY primary_class
ORDER BY n DESC;

-- @title: 6. 运行成本 / 延迟 / judge 覆盖率（同集合决策样本）
SELECT run_id, run_type, n, split,
       ROUND(total_cny, 4) AS total_cny,
       ROUND(per_item_cny, 4) AS per_item_cny,
       ROUND(p50_ms, 0) AS p50_ms,
       ROUND(p95_ms, 0) AS p95_ms,
       ROUND(judge_coverage, 4) AS judge_coverage,
       ROUND(ac, 4) AS answer_correctness,
       ROUND(false_refusal, 4) AS false_refusal,
       ROUND(recall5, 4) AS recall_at_5
FROM runs
WHERE run_id IN ('20261013-003', '20261013-004', '20261013-005', '20261016-002', '20261019-001')
   OR (run_id = '20261014-004' AND run_type = 'acceptance')
ORDER BY run_id;
