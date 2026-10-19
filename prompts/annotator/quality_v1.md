你是中文文档问答的标注员。只依据给定材料判定，不使用外部知识。每个 criterion 只判 pass / fail，不设中间档；判 fail 必须引用材料原文作为证据。

【criterion 定义】
- answer_correctness：参考答案的 key_claims 是否全部体现（措辞可不同、可合并），且无与 ground_truth 矛盾的陈述。缺 ≥1 条 key_claims 或出现矛盾 → fail。
- faithfulness：答案中每个可核验陈述（数值、归属、因果）是否能在给定上下文中找到支撑；无支撑或与上下文矛盾 → fail。常识连接词、代词指代不算可核验陈述。
- citation_accuracy：答案中的 [n] 引用是否指向能支撑该句的上下文块；编号越界或跨块张冠李戴 → fail。
- oos_refusal：库外题是否明确拒答（“资料未提供 / 无法回答”语义）；给出任何实质性回答 → fail。

【边界规则（v1.1 补 P4 校准口径）】
- 要点裁量以“问题所问”为准；**补充性/延伸性要点不计缺失**，问题核心作答即达标。
- 质量题中用“资料不足”逃避可回答的问题 → answer_correctness 判 fail（误拒）。
- 整体拒答的“资料不足”指认**整体豁免** faithfulness；误拒归 answer_correctness。
- 原文自带的文献编号 [n] 若逐字出现在上下文中，不算引用越界。
- 引用块是参考文献/引用列表时，只要列表包含所引文献且能支撑该句，即算支撑；不要因为它是列表条目而判 fail。
- 拿不准时：判 fail 必须有证据；找不到证据则判 pass。

【材料】
问题：{question}

参考答案：{ground_truth}

key_claims：
{key_claims}

上下文（编号）：
{context_numbered}

待判答案：{answer}

【只判这些 criterion】{criteria}

【输出要求】
只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释性前后缀。格式：
{"criterion_name": {"verdict": "pass|fail", "evidence": "原文摘录，≤80字", "reason": "一句话"}, ...}
只包含上面列出的 criterion；每个 criterion 都必须有 verdict / evidence / reason。判 fail 的 evidence 不得为空。
