from __future__ import annotations

import json
import os
import pathlib
import re

from openai import OpenAI

from summarizer import (MapSummarizer, build_global_context, _parse_llm_json,
                        normalize_numbers, format_number)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
REDUCE_VERSION = "r2"   

REDUCE_SYSTEM_PROMPT = """你是一名计算机学术论文总结专家。你将收到：
各章节的分段要点、必须保留的数字清单（每条带归属：谁的什么指标、在什么条件下）、
解析得到的表格原文（行列结构），以及公式/伪代码的结论条目。
请把它们整合成一篇高质量的中文总结。

【什么算高质量】
1. 提炼一条主线：先从材料中判断这篇论文要解决的核心问题和核心主张
   （例如"推翻某已有假设，用新表示换来多数场景的提升"），
   全文围绕主线组织，让读者读完能一句话复述论文的贡献；
2. 分层与详略：与主线直接相关的写透，背景知识（如某模型的基础原理）、
   过程性细节（如某步骤的可视化安排）一句带过或不写；
   方法是论文的核心：方法部分的叙述篇幅应约为背景/设置部分的两倍，
   与结论无关的训练细节（初始化方式、激活函数、优化器名等）删去；
   实验数据表格不计入叙述篇幅；
3. 严谨措辞：指标名与表述一致（AUC 不是"准确率"）；
   材料中缺少上下文的引用（如"与文献 [15] 不同"而无具体内容）不要硬写；
4. 局限部分必须具体：优先覆盖材料中出现的失败/反例场景及其原因，
   不只写"留作未来工作"。

【硬性要求】
5. 必须保留数字清单里的所有数字，并按清单标注的归属正确表述，
   禁止改成"显著提升"这类模糊表述，禁止把数字安到错误的方法/数据集上；
6. 表格原文是解析得到的可靠数据：实验结果类表格（如各方法 AUC 对比）
   的数值必须完整整理进总结，可用 Markdown 表格呈现，
   数值照抄原文，禁止换算、四舍五入或只写"优于"不写数值；
7. 专有名词、数据集名、公式符号保留英文原文；
8. 只使用给定材料中的信息，不要补充外部知识，不要臆造对比；
9. 输出连贯的段落式总结，可用小标题，篇幅 600~1000 字。"""

REDUCE_USER_PROMPT = """【各段要点】（按论文顺序）
{key_points}

【表格原文】（解析直通，数值可靠，实验结果表格必须完整呈现）
{tables}

【表格/公式/伪代码结论】
{assets}

【必须包含的数字】（括号内是归属与条件，写进总结时必须对应正确）
{numbers}

请输出最终总结。"""

_NUM_TOKEN = re.compile(r"\d+\.?\d*")

CRITIC_SYSTEM_PROMPT = """你是学术论文总结的质检员（critic）。你会收到总结所依据的原始材料
（分段要点、表格原文、数字清单）和一稿总结草稿。你的任务不是重写，而是对照材料挑出草稿的具体问题。

逐项检查：
1. 归属错误：数字清单或表格里的数值被安到错误的方法、数据集或场景上；
2. 幻觉与硬写：出现了材料中没有的信息；对缺少上下文的引用（如"与文献 [15] 不同"
   但材料没说 [15] 是什么）做了具体展开；
3. 主线：读完能否一句话复述论文的核心贡献？主线不清是重大问题；
4. 冗余罗列：与结论无关的参数、过程性细节（列了但支撑不了任何结论）；
5. 结构游离：某段内容放在了错误位置（如相关工作对比被当作结尾）；
6. 措辞不严谨：指标名与表述不一致（如把 AUC 说成"准确率"）、用模糊词替代数字。

只报告有把握的问题，不吹毛求疵；没有问题就 pass。
你必须只输出一个 JSON 对象，格式二选一：
{"pass": true, "issues": []}
{"pass": false, "issues": [{"type": "归属错误|幻觉硬写|主线|冗余|结构|措辞", "detail": "问题描述（引用草稿原句）", "fix": "具体修改建议"}]}"""

CRITIC_USER_PROMPT = """【总结所依据的材料】

【各段要点】
{key_points}

【表格原文】
{tables}

【公式/伪代码结论】（草稿中的公式与算法细节应能在这些结论中找到出处）
{assets}

【数字清单】（每条带归属）
{numbers}

【待审草稿】
{draft}

请审查并输出 JSON。"""

def render_tables(tables: list[dict]) -> str:
    """parser 直通的表格原文按 section 拼段。tables 来自 parse() 返回，不经过 LLM。"""
    parts = []
    for t in tables or []:
        text = (t.get("text") or "").strip()
        if text:
            parts.append(f"[{t.get('section') or '（无章节）'}]\n{text}")
    return "\n\n".join(parts)

def assemble_reduce_input(summaries: list[dict]) -> dict:
    """map 摘要装配：要点按序、assets 去重、numbers 归一去重取并集。"""
    kp_parts: list[str] = []
    seen_numbers: dict = {}      # canonical json -> record，保序去重
    seen_assets: dict = {}       # (type, desc) -> None，保序去重
    for s in summaries:
        sec = s.get("section") or f"chunk#{s.get('chunk_id')}"
        pts = "\n".join(f"- {p}" for p in s.get("key_points", []))
        if pts:
            kp_parts.append(f"[{sec}]\n{pts}")
        for n in normalize_numbers(s.get("numbers", [])):
            key = json.dumps(n, ensure_ascii=False, sort_keys=True)
            seen_numbers.setdefault(key, n)
        for a in s.get("assets", []):
            desc = (a.get("desc") or "").strip()
            if not desc or "未出现" in desc:   # 滤占位废话
                continue
            seen_assets.setdefault((a.get("type"), desc), None)
    return {
        "key_points_text": "\n\n".join(kp_parts),
        "assets_text": "\n".join(f"- [{t}] {d}" for (t, d) in seen_assets),
        "numbers": list(seen_numbers.values()),
    }

def check_numbers(final_text: str, numbers: list[dict]) -> list[str]:
    """数字守恒：每条记录 value 里的数值 token 都出现在最终总结里才算有。
    判定刻意宽松，避免"λ1=300"写成"λ1 为 300"就误报；返回缺失记录的可读串。"""
    missing = []
    for n in numbers:
        vals = _NUM_TOKEN.findall(n.get("value", ""))
        if vals and all(v in final_text for v in vals):
            continue
        missing.append(format_number(n))
    return missing

class Reducer:
    def __init__(self, client: OpenAI | None = None, temperature: float = 0.3):
        # client 允许外部注入（测试时塞桩对象）；默认按 DeepSeek 兼容模式构造
        self.client = client or OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
        )
        self.model = DEEPSEEK_MODEL
        self.temperature = temperature

    def _call_llm(self, prompt: str, system: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _critic(self, asm: dict, tables_text: str, draft: str) -> list[dict]:
        """质量审查：critic 对照材料给 draft 挑刺，返回 issues 列表（空 = 通过）。
        critic 本身失败（网络/JSON 坏）时降级为无意见，守恒校验仍兜底。"""
        prompt = CRITIC_USER_PROMPT.format(
            key_points=asm["key_points_text"],
            tables=tables_text or "（无）",
            assets=asm["assets_text"] or "（无）",
            numbers="\n".join(f"- {format_number(n)}" for n in asm["numbers"]) or "（无）",
            draft=draft,
        )
        try:
            verdict = _parse_llm_json(
                self._call_llm(prompt, CRITIC_SYSTEM_PROMPT, max_tokens=1000))
            if isinstance(verdict, dict) and not verdict.get("pass"):
                return [i for i in verdict.get("issues", []) if isinstance(i, dict)]
            return []
        except Exception:
            return []
    
    def reduce_all(self, summaries: list[dict], tables: list[dict]) -> dict:
        """装配材料 → 一次 LLM 调用产出最终总结 → 数字守恒校验，缺失时补救一次"""
        asm = assemble_reduce_input(summaries)
        tables_text = render_tables(tables)
        user = REDUCE_USER_PROMPT.format(
            key_points=asm["key_points_text"],
            tables=tables_text or "（无）",
            assets=asm["assets_text"] or "（无）",
            numbers="\n".join(f"- {format_number(n)}" for n in asm["numbers"]) or "（无）",
        )

        draft = self._call_llm(user, REDUCE_SYSTEM_PROMPT, max_tokens=2000)
        missing = check_numbers(draft, asm["numbers"])
        issues = self._critic(asm, tables_text, draft)

        if missing or issues:
            extra = ""
            if missing:
                extra += ("\n\n【数字缺失】以下带归属的数字在草稿中缺失，"
                            "必须全部体现（归属必须对应正确）：\n"
                            + "\n".join(f"- {m}" for m in missing))
            if issues:
                extra += ("\n\n【审查意见】逐条修复，每条必须实际落实到文字，"
                            "修订完成后自查各条是否仍存在；其余内容保持不变：\n"
                            + "\n".join(f"- {i.get('detail')} → 修法：{i.get('fix')}"
                                        for i in issues))
            final = self._call_llm(user + extra, REDUCE_SYSTEM_PROMPT, max_tokens=2000)
            missing_final = check_numbers(final, asm["numbers"])
            if len(missing_final) > len(missing):    # 修订稿守恒回退 → 保留原稿
                final, missing_final = draft, missing
        else:
            final, missing_final = draft, missing

        return {"summary": final,
                "missing_numbers": missing_final,
                "numbers_total": len(asm["numbers"]),
                "critic_issues": issues}

    
if __name__ == "__main__":
    from parser import DoclingParser
    from chunker import MapReduceChunker

    PDF = r"D:\githubProject\document summarize  agent\test doc\Clustering_based_Autoencoder_for_Anomaly_Detection1.pdf"

    data = DoclingParser().load_or_parse(PDF)          # 解析缓存命中，秒回
    chunks = MapReduceChunker().chunk(data["blocks"])
    context = build_global_context(data["blocks"])
    print("全局上下文：\n" + context[:200] + "...\n")

    if "DEEPSEEK_API_KEY" not in os.environ:
        raise SystemExit("未检测到 DEEPSEEK_API_KEY。PowerShell 里先执行：\n"
                         '  $env:DEEPSEEK_API_KEY="sk-你的key"\n再重跑本脚本。')

    # map：全部走 .summary_cache/，之前跑过的 chunk 秒回不花钱
    summarizer = MapSummarizer(max_workers=4)
    summaries = summarizer.map_all(chunks, context)

    # reduce：map 摘要 + parse 表格原文直通（不传 context，见 reduce_all docstring）
    reducer = Reducer()
    result = reducer.reduce_all(summaries, data["tables"])

    print("=" * 72 + "\n最终总结\n" + "=" * 72)
    print(result["summary"])
    if result["critic_issues"]:
        print(f"\ncritic 提出 {len(result['critic_issues'])} 条意见（已修订）：")
        for i in result["critic_issues"]:
            print(f" - [{i.get('type')}] {i.get('detail')}")
    else:
        print("\ncritic: 一稿通过，无意见")
    ok = result["numbers_total"] - len(result["missing_numbers"])
    print(f"\n数字守恒: {ok}/{result['numbers_total']}")
    if result["missing_numbers"]:
        print("缺失:", "；".join(result["missing_numbers"]))

    out = pathlib.Path(__file__).with_name("final_summary.md")
    out.write_text(result["summary"], encoding="utf-8")
    print(f"(已写入 {out.name})")
