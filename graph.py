from __future__ import annotations

import operator
import threading
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from openai import OpenAI

from summarizer import (MapSummarizer, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
                        build_global_context, normalize_numbers, format_number,
                        _is_junk, _parse_llm_json)
from reducer import Reducer, check_numbers, render_tables
from chat import build_materials, save_summary_md
from parser import DoclingParser
from chunker import MapReduceChunker

STUFF_THRESHOLD_TOKENS = 2000     # 全文低于此 token 数走 stuff 单次总结
MAP_CONCURRENCY = 4               # Send 分支的最大并发（信号量）

_MAP_SEM = threading.Semaphore(MAP_CONCURRENCY)

STUFF_SYSTEM_PROMPT = """你是一名计算机学术论文总结专家。请阅读整篇论文材料，
直接产出最终总结和其中必须保留的数字清单。

【什么算高质量】
1. 提炼一条主线：先判断论文要解决的核心问题和核心主张，全文围绕主线组织，
   读者读完能一句话复述论文的贡献；
2. 分层与详略：与主线直接相关的写透，背景知识、过程性细节一句带过或不写；
   方法是论文的核心：方法叙述篇幅应约为背景/设置部分的两倍；
3. 严谨措辞：指标名与表述一致（AUC 不是"准确率"）；缺少上下文的引用不硬写；
4. 局限部分必须具体：优先覆盖失败/反例场景及其原因。

【硬性要求】
5. numbers 收录正文中零散出现的关键数值（超参数、实验设置等），每条一个对象：
   value 原样必填；metric 指标名；subject 是谁的数值；condition 在什么条件下；
   判断不出填 ""。禁止把符号名填进 value，禁止收文献引用/公式/图表编号；
6. 表格数值不进 numbers（表格原文会另行直通），但实验结果表格必须完整整理进
   summary，可用 Markdown 表格，数值照抄禁止换算；
7. 专有名词、数据集名保留英文；只使用材料中的信息；
8. summary 为连贯段落式中文总结，可用小标题，篇幅 600~1000 字。

【输出格式】只输出 JSON 对象：
{"summary": "最终总结全文",
 "numbers": [{"metric": "λ1", "value": "300", "subject": "CAE 损失函数", "condition": ""}]}"""

STUFF_USER_PROMPT = """【论文全局信息】
{global_context}

【论文全文材料】
{materials}

【表格原文】（解析直通，数值可靠，实验结果表格必须完整呈现进 summary）
{tables}

请输出 JSON。"""


class PaperState(TypedDict):
    pdf_path: str
    data: dict                                        # parse 结果
    context: str                                      # build_global_context 产出
    chunks: list                                      # chunker 产出
    summaries: Annotated[list, operator.add]          # Send 分支结果自动聚合
    final: dict                                       # {summary, missing_numbers, ...}

def build_graph(client: OpenAI):
    """所有节点用闭包共享同一个 client/summarizer/reducer，图定义编译一次。"""
    summarizer = MapSummarizer(client=client, max_workers=1)   # 并发由信号量管
    reducer = Reducer(client=client)

    # ---------------- 节点 ----------------
    def parse_node(state: PaperState) -> dict:
        data = DoclingParser().load_or_parse(state["pdf_path"])   # 解析缓存
        return {"data": data, "context": build_global_context(data["blocks"])}

    def route(state: PaperState) -> str:
        text = "\n".join((b.get("text") or "") for b in state["data"]["blocks"])
        tokens = len(text) // 3                        # 粗估
        return "stuff" if tokens <= STUFF_THRESHOLD_TOKENS else "mapreduce"

    def chunk_node(state: PaperState) -> dict:
        return {"chunks": MapReduceChunker().chunk(state["data"]["blocks"])}

    def fan_out(state: PaperState) -> list[Send]:
        """按 chunk 数动态派工：每个非垃圾 chunk 一张 Send 派工单。"""
        return [Send("map_one", {"chunk": c, "context": state["context"]})
                for c in state["chunks"] if not _is_junk(c)]

    def map_one_node(payload: dict) -> dict:
        """Send 分支的执行体：单 chunk 的 map（带缓存 + 重试兜底）。
        并发上限由 _MAP_SEM 控制，所有分支共享同一把信号量。"""
        with _MAP_SEM:
            s = summarizer._map_with_retry(payload["chunk"], payload["context"])
        return {"summaries": [s]}

    def reduce_node(state: PaperState) -> dict:
        ordered = sorted(state["summaries"], key=lambda s: s.get("chunk_id", 0))
        result = reducer.reduce_all(ordered, state["data"]["tables"])
        result["route"] = "mapreduce"
        return {"final": result}

    def stuff_node(state: PaperState) -> dict:
        """短文档：全文一次调用出 {summary, numbers}，守恒校验 + 补救一次。
        输出结构与 reduce_node 对齐，下游不关心走哪条路。"""
        data, context = state["data"], state["context"]
        user = STUFF_USER_PROMPT.format(
            global_context=context,
            materials=build_materials(data["blocks"]),
            tables=render_tables(data["tables"]) or "（无）",
        )

        def _call(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": STUFF_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=4000,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content.strip()

        parsed = _parse_llm_json(_call(user))
        summary = parsed.get("summary", "")
        numbers = normalize_numbers(parsed.get("numbers", []))
        missing = check_numbers(summary, numbers)
        if missing:                                     # 缺失带清单补救一次
            user2 = user + ("\n\n【数字缺失】以下带归属的数字在上一稿中缺失，"
                            "必须全部体现（归属对应正确），summary 其余内容保持不变：\n"
                            + "\n".join(f"- {m}" for m in missing))
            parsed2 = _parse_llm_json(_call(user2))
            s2 = parsed2.get("summary", "")
            m2 = check_numbers(s2, numbers)
            if len(m2) < len(missing):                  # 补救稿更好才采纳
                summary, missing = s2, m2
        return {"final": {"summary": summary, "missing_numbers": missing,
                          "numbers_total": len(numbers), "route": "stuff"}}

    g = StateGraph(PaperState)
    g.add_node("parse", parse_node)
    g.add_node("chunk", chunk_node)
    g.add_node("map_one", map_one_node)
    g.add_node("reduce", reduce_node)
    g.add_node("stuff", stuff_node)
    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", route, {"mapreduce": "chunk", "stuff": "stuff"})
    g.add_conditional_edges("chunk", fan_out, ["map_one"])
    g.add_edge("map_one", "reduce")
    g.add_edge("reduce", END)
    g.add_edge("stuff", END)
    return g.compile()

if __name__ == "__main__":
    import os

    PDF = r"D:\githubProject\document summarize  agent\test doc\Clustering_based_Autoencoder_for_Anomaly_Detection1.pdf"

    if "DEEPSEEK_API_KEY" not in os.environ:
        raise SystemExit("未检测到 DEEPSEEK_API_KEY。PowerShell 里先执行：\n"
                         '  $env:DEEPSEEK_API_KEY="sk-你的key"\n再重跑本脚本。')
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=DEEPSEEK_BASE_URL)

    app = build_graph(client)
    result = app.invoke({"pdf_path": PDF})
    final = result["final"]

    print("=" * 72)
    print(f"[路由: {final['route']}]")
    print(final["summary"])
    ok = final["numbers_total"] - len(final["missing_numbers"])
    print(f"\n数字守恒: {ok}/{final['numbers_total']}")
    if final["missing_numbers"]:
        print("缺失:", "；".join(final["missing_numbers"]))
    md = save_summary_md(PDF, final["summary"],
                         final["numbers_total"], len(final["missing_numbers"]))
    print(f"(总结已写入 {md.name}；立即重跑本脚本，LLM 节点应全命中缓存，只剩 reduce 2~3 次调用)")