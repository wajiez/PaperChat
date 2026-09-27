from __future__ import annotations

import os
import re
import time
import hashlib
import pathlib
import json

from openai import OpenAI

SUMMARY_CACHE_DIR = pathlib.Path(__file__).parent / ".summary_cache"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

PROMPT_VERSION = "v7"

MAP_SYSTEM_PROMPT = (
    "你是一个学术论文分段摘要器。只依据给定内容写摘要，不臆造、不补充原文没有的信息。"
    "你必须只输出一个合法的 JSON 对象。"
)

MAP_USER_PROMPT = """下面是一篇论文的一个片段，你负责总结这一片段（不是全文）。

【论文全局信息】
{global_context}

【当前片段】（章节：{section}）
{chunk_text}

【要求】
1. key_points：4~8 条要点，每条一句话；
   方法/机制/流程类内容必须拆条逐步说明——每个步骤或组成部分单独一条，
   禁止把多步机制压缩成一句话；
2. 必须保留具体事实：模型名、数据集名、超参数、实验数字；
3. numbers：正文里零散出现的数值，每条一个对象：
   - "value"：原样数值，必填，不得四舍五入、不得改写；
   - "metric"：指标名（如 "AUC"、"λ1"）；
   - "subject"：这是谁的数值（如 "CAE (M-CEN)"），判断不出填 ""；
   - "condition"：在什么数据集/场景/设置下，判断不出填 ""；
   - 禁止把符号名（如 λ1）填进 value；文字化数量词（如 four scenarios）不收；
   - 不收：文献引用编号（[11]）、公式/图表/章节编号、页码——它们是位置标记不是实验数值；
4. 表格数值不用抄进 numbers，表的存在和结论在 assets 的 desc 里概括即可;
5. 片段中的公式/伪代码在 assets 里各占一条，type 取 formula/code，
   desc 用 2~3 句说明：定义了什么、各项/各步的含义、在整体方法中的作用，
   关键符号保留原文；没有则不输出该类，禁止输出"未出现"之类的占位描述；
6. 不要写"本片段介绍了"这类元话语，直接陈述内容；
7. 只总结【当前片段】中实际出现的内容，全局信息仅用于理解背景，禁止将其展开为要点；
8. key_points、note、assets 的 desc 一律用中文，
   专有名词（CAE、SAE、CTU13 等）、公式符号、数据集名保留原文。

【输出格式】只输出 JSON 对象，不要输出任何其他文字：
{{
  "key_points": ["要点1", "要点2"],
  "numbers": [
    {{"metric": "AUC", "value": "0.996", "subject": "CAE (M-CEN)", "condition": "CTU13-10"}},
    {{"metric": "λ1", "value": "300", "subject": "CAE 损失函数", "condition": ""}}
  ],
  "tables": [
    {{"caption": "表1 数据集统计", "note": "四个数据集的维度与样本量",
      "columns": ["维度", "训练集", "正常测试", "异常测试"],
      "rows": [{{"row": "Rbot (CTU13-10)", "values": ["38", "6338", "9509", "63812"]}}]}}
  ],
  "assets": [{{"type": "formula", "desc": "..."}}]
}}
"""

_NUM_TOKEN = re.compile(r"\d+\.?\d*")
_IDENT_DIGIT = re.compile(r"(?<=[A-Za-z\u00c0-\u024f\u0370-\u03ff\u4e00-\u9fff])\d")

def build_global_context(blocks: list[dict], max_chars: int = 1200) -> str:
    """从 blocks 提取论文标题 + 摘要节选，供每个 map 请求当全局上下文。"""
    title = next((b["text"] for b in blocks if b["type"] == "heading"), "")
    abstract = ""
    for b in blocks:
        if b["type"] == "prose" and "Abstract" in b["text"]:
            abstract = b["text"]
            break
    return f"论文标题：{title}\n摘要节选：{abstract[:max_chars]}"

def _parse_llm_json(text: str) -> dict:
    """三级降级：严格 → 宽松(strict=False) → json-repair 修复"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("响应中找不到 JSON 对象")
    body = text[start:end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(body, strict=False)   # 容忍字符串里的裸换行/控制字符
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json      # 兜底：修复未转义引号等
    except ImportError:
        raise ValueError(
            "JSON 解析失败且未安装 json-repair，请先执行：pip install json-repair")
    return json.loads(repair_json(body))

_JUNK_WORDS = ("reference", "bibliograph", "acknowledg", "致谢", "鸣谢",
               "funding", "资助", "declaration", "appendix", "附录",
               "参考文献", "目录")

def _is_junk(c):
    s = (c.get("section") or "").lower().replace(" ", "").replace("　", "")
    if any(w in s for w in _JUNK_WORDS):
        return True
    return not c.get("section") and c["tokens"] < 100

def normalize_numbers(raw) -> list[dict]:
    """把 LLM 返回的 numbers 归一成 [{metric,value,subject,condition}]"""
    out: list[dict] = []
    for item in raw or []:
        if isinstance(item, str):
            rec = {"value": item.strip()}
        elif isinstance(item, dict):
            rec = item
        else:
            continue
        v = str(rec.get("value") or "").strip()
        # 剔除标识符相邻数字（λ1 的 1、x2 的 2）后仍要有数字 token，否则丢弃
        if not v or not re.search(r"\d+\.?\d*", _IDENT_DIGIT.sub("", v)):
            continue
        out.append({
            "value": v,
            "metric": rec.get("metric") if isinstance(rec.get("metric"), str) else "",
            "subject": rec.get("subject") if isinstance(rec.get("subject"), str) else "",
            "condition": rec.get("condition") if isinstance(rec.get("condition"), str) else "",
        })
    return out

def format_number(n: dict) -> str:
    """渲染成可读串：`AUC=0.996（CAE(M-CEN)，CTU13-10）`"""
    head = f"{n['metric']}={n['value']}" if n.get("metric") else n["value"]
    notes = [p for p in (n.get("subject"), n.get("condition")) if p]
    return f"{head}（{'，'.join(notes)}）" if notes else head

def _filter_assets(assets: list[dict], chunk: dict) -> list[dict]:
    """幻觉过滤：asset 的 type 必须是该 chunk 真实包含的块类型。
    chunker 打包时写下的 types 来自真实 blocks，模型报告了 chunk 里不存在的
    类型（如无代码块却报 [code] 伪代码）即为幻觉，直接丢弃。"""
    types = set(chunk.get("types") or [])
    if not types:
        return assets
    return [a for a in assets if a.get("type") in types]

class MapSummarizer:
    def __init__(self, client: OpenAI | None = None,
                 max_workers: int = 4, max_retries: int = 1,
                 temperature: float = 0.3, max_tokens: int = 1500,
                 use_cache: bool = True):
        # client 允许外部注入
        self.client = client or OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
        )
        self.model = DEEPSEEK_MODEL
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cache = use_cache

    # ---------------- 缓存 ----------------
    def _cache_key(self, chunk: dict, global_context: str) -> str:
        """chunk 内容 + 全局上下文 + prompt 版本 + 模型/温度，任一变化缓存即失效。"""
        h = hashlib.sha1(json.dumps([
            PROMPT_VERSION, self.model, self.temperature,
            global_context, chunk["text"],
        ], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        return f"c{chunk['chunk_id']}_{h}"

    def _cache_load(self, key: str) -> dict | None:
            f = SUMMARY_CACHE_DIR / f"{key}.json"
            if self.use_cache and f.exists():
                return json.loads(f.read_text(encoding="utf-8"))
            return None

    def _cache_save(self, key: str, result: dict) -> None:
            if not self.use_cache or result.get("error"):
                return
            SUMMARY_CACHE_DIR.mkdir(exist_ok=True)
            (SUMMARY_CACHE_DIR / f"{key}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
            
    # ---------------- 单个 map ----------------
    def _call_llm(self, chunk: dict, global_context: str) -> dict:
            prompt = MAP_USER_PROMPT.format(
                global_context=global_context,
                section=chunk.get("section") or "（无章节信息）",
                chunk_text=chunk["text"],
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MAP_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},  # DeepSeek 的 JSON 模式
            )
            return _parse_llm_json(resp.choices[0].message.content)
    
    def map_one(self, chunk: dict, global_context: str) -> dict:
        key = self._cache_key(chunk, global_context)
        
        cached = self._cache_load(key)
        if cached is not None:  # 命中缓存：不花钱不等待
            return {**cached, "chunk_id": chunk["chunk_id"], "cached": True}

        data = self._call_llm(chunk, global_context)
        result = {
            "chunk_id": chunk["chunk_id"],
            "section": chunk.get("section") or "",
            "key_points": data.get("key_points", []),
            "numbers": normalize_numbers(data.get("numbers", [])),
            "assets":  _filter_assets(data.get("assets", []), chunk),
        }
        self._cache_save(key, result)
        return result

    def _map_with_retry(self, chunk: dict, global_context: str) -> dict:
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                return self.map_one(chunk, global_context)
            except Exception as e:  # 网络/限流/超时/JSON 解析失败等，重试一次再放弃
                last_err = e
                time.sleep(2)
        return {
            "chunk_id": chunk["chunk_id"],
            "section": chunk.get("section") or "",
            "key_points": [], "numbers": [], "assets": [],
            "error": f"{type(last_err).__name__}: {last_err}",
        }

    # ---------------- 并行 map ----------------
    def map_all(self, chunks: list[dict], global_context: str,
                limit: int | None = None) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        todo = [c for c in chunks if not _is_junk(c)]
        todo = todo[:limit] if limit else todo

        results: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._map_with_retry, c, global_context): c["chunk_id"]
                for c in todo
            }
            for fut in as_completed(futures):
                cid = futures[fut]
                results[cid] = fut.result()

        return [results[c["chunk_id"]] for c in todo]  # 按原顺序还给 reduce

if __name__ == "__main__":
    from parser_wb import DoclingParserWB
    from chunker import MapReduceChunker

    PDF = r"D:\githubProject\document summarize  agent\test doc\Clustering_based_Autoencoder_for_Anomaly_Detection1.pdf"

    data = DoclingParserWB().load_or_parse(PDF)          # 缓存命中，秒回
    chunks = MapReduceChunker().chunk(data["blocks"])
    context = build_global_context(data["blocks"])
    print("全局上下文：\n" + context[:200] + "...\n")

    if "DEEPSEEK_API_KEY" not in os.environ:
        raise SystemExit("未检测到 DEEPSEEK_API_KEY。PowerShell 里先执行：\n"
                         '  $env:DEEPSEEK_API_KEY="sk-你的key"\n再重跑本脚本。')

    summarizer = MapSummarizer(max_workers=4)
    summaries = summarizer.map_all(chunks, context)

    for s in summaries: # 第二次跑同一路径：全部走 .summary_cache/
        print("=" * 72)
        print(f"[#{s['chunk_id']} | {s['section']}]"
                + (" [缓存]" if s.get("cached") else "")
                + (" ⚠ " + s["error"] if s.get("error") else ""))
        for p in s["key_points"]:
            print(" •", p)
        for n in s["numbers"]:
            print(" #", format_number(n))
        for t in s.get("tables", []):
            print(f" [table] {t['caption']}（{t['note']}）")
            print("   列:", " | ".join(t["columns"]))
            for r in t["rows"]:
                print(f"   {r['row']}: " + " | ".join(r["values"]))
        for a in s["assets"]:
            print(f" [{a.get('type')}]", a.get("desc", ""))

