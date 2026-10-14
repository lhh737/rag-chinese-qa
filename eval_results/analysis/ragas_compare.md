# RAGAS 对照（faithfulness，附录级）

- 样本：校准集答案类 52 题（与人工盲标/自研 judge 同题同答案）
- RAGAS 0.4.3 · LLM backend: deepseek-flash（官方）· 并发 6

## 主口径（score==1.0 → pass）

- RAGAS vs 人工：26/52 = 0.500
- RAGAS vs 自研 judge：26/52 = 0.500
- 与人工分歧 26 条：q001(人工pass/RAGASfail@0.0), q003(人工fail/RAGASpass@1.0), q012(人工pass/RAGASfail@0.0), q016(人工pass/RAGASfail@0.0), q017(人工pass/RAGASfail@0.75), q018(人工pass/RAGASfail@0.0), q031(人工pass/RAGASfail@0.0), q032(人工pass/RAGASfail@0.969), q033(人工pass/RAGASfail@0.923), q034(人工pass/RAGASfail@0.0), q035(人工pass/RAGASfail@0.5), q037(人工pass/RAGASfail@0.0), q038(人工pass/RAGASfail@0.0), q039(人工pass/RAGASfail@0.0), q040(人工pass/RAGASfail@0.0), q041(人工pass/RAGASfail@0.857), q046(人工pass/RAGASfail@0.857), q067(人工pass/RAGASfail@0.833), q068(人工pass/RAGASfail@0.667), q069(人工pass/RAGASfail@0.857), q070(人工pass/RAGASfail@0.0), q072(人工pass/RAGASfail@0.75), q073(人工pass/RAGASfail@0.75), q074(人工pass/RAGASfail@0.889), q075(人工pass/RAGASfail@0.857), q076(人工pass/RAGASfail@0.6)
- 与 judge 分歧 26 条：q001(judgepass/RAGASfail@0.0), q003(judgefail/RAGASpass@1.0), q012(judgepass/RAGASfail@0.0), q016(judgepass/RAGASfail@0.0), q017(judgepass/RAGASfail@0.75), q018(judgepass/RAGASfail@0.0), q031(judgepass/RAGASfail@0.0), q032(judgepass/RAGASfail@0.969), q033(judgepass/RAGASfail@0.923), q034(judgepass/RAGASfail@0.0), q035(judgepass/RAGASfail@0.5), q037(judgepass/RAGASfail@0.0), q038(judgepass/RAGASfail@0.0), q039(judgepass/RAGASfail@0.0), q040(judgepass/RAGASfail@0.0), q041(judgepass/RAGASfail@0.857), q046(judgepass/RAGASfail@0.857), q067(judgepass/RAGASfail@0.833), q068(judgepass/RAGASfail@0.667), q069(judgepass/RAGASfail@0.857), q070(judgepass/RAGASfail@0.0), q072(judgepass/RAGASfail@0.75), q073(judgepass/RAGASfail@0.75), q074(judgepass/RAGASfail@0.889), q075(judgepass/RAGASfail@0.857), q076(judgepass/RAGASfail@0.6)

## 敏感性（>=0.8 → pass）

- RAGAS vs 人工：34/52 = 0.654
- RAGAS vs 自研 judge：34/52 = 0.654
- 与人工分歧 18 条：q001(人工pass/RAGASfail@0.0), q003(人工fail/RAGASpass@1.0), q012(人工pass/RAGASfail@0.0), q016(人工pass/RAGASfail@0.0), q017(人工pass/RAGASfail@0.75), q018(人工pass/RAGASfail@0.0), q031(人工pass/RAGASfail@0.0), q034(人工pass/RAGASfail@0.0), q035(人工pass/RAGASfail@0.5), q037(人工pass/RAGASfail@0.0), q038(人工pass/RAGASfail@0.0), q039(人工pass/RAGASfail@0.0), q040(人工pass/RAGASfail@0.0), q068(人工pass/RAGASfail@0.667), q070(人工pass/RAGASfail@0.0), q072(人工pass/RAGASfail@0.75), q073(人工pass/RAGASfail@0.75), q076(人工pass/RAGASfail@0.6)
- 与 judge 分歧 18 条：q001(judgepass/RAGASfail@0.0), q003(judgefail/RAGASpass@1.0), q012(judgepass/RAGASfail@0.0), q016(judgepass/RAGASfail@0.0), q017(judgepass/RAGASfail@0.75), q018(judgepass/RAGASfail@0.0), q031(judgepass/RAGASfail@0.0), q034(judgepass/RAGASfail@0.0), q035(judgepass/RAGASfail@0.5), q037(judgepass/RAGASfail@0.0), q038(judgepass/RAGASfail@0.0), q039(judgepass/RAGASfail@0.0), q040(judgepass/RAGASfail@0.0), q068(judgepass/RAGASfail@0.667), q070(judgepass/RAGASfail@0.0), q072(judgepass/RAGASfail@0.75), q073(judgepass/RAGASfail@0.75), q076(judgepass/RAGASfail@0.6)
