from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pathlib
import sys
import threading
import time

from openai import OpenAI

from summarizer import DEEPSEEK_BASE_URL
from chat import save_summary_md
from graph import build_graph

BATCH_CONCURRENCY = 2                                        # 同时在跑的论文数
BATCH_DIR = pathlib.Path(__file__).parent / "test docs"      # 缺省 PDF 目录
STATE_DIR = pathlib.Path(__file__).parent / ".batch_state"   # 断点续跑状态
REPORT_PATH = pathlib.Path(__file__).parent / "batch_report.md"

_PRINT_LOCK = threading.Lock()   # 多篇并行打印不串行

# ---------------- 断点续跑状态 ----------------
def _state_path(pdf: pathlib.Path) -> pathlib.Path:
    key = hashlib.sha1(str(pdf.resolve()).encode("utf-8")).hexdigest()[:12]
    return STATE_DIR / f"{key}.json"


def _load_state(pdf: pathlib.Path) -> dict:
    p = _state_path(pdf)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}          # 状态文件坏了当没跑过，重跑由缓存兜底
    return {}

# ---------------- 单篇执行 ----------------
def _save_state(pdf_file: pathlib.Path, **fields) -> None:
    rec = _load_state(pdf_file)
    rec.update(fields)
    _state_path(pdf_file).write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                encoding="utf-8")
def run_one(pdf: pathlib.Path, idx: int, total: int) -> dict:
    tag = f"[{idx}/{total}] {pdf.name}"

    state = _load_state(pdf)
    if state.get("status") == "done":
        with _PRINT_LOCK:
            print(f"[跳过] {tag}（已完成，守恒 {state.get('conservation', '?')}）")
        rec = dict(state)
        rec["skipped"] = True
        return rec

    # running / failed / 没跑过 → 都重跑（缓存兜底）
    _save_state(pdf, status="running", pdf=pdf.name)
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=DEEPSEEK_BASE_URL)
    app = build_graph(client)
    t0 = time.time()
    with _PRINT_LOCK:
        print(f"[开始] {tag}")
    try:
        final = app.invoke({"pdf_path": str(pdf)})["final"]
        ok = final["numbers_total"] - len(final["missing_numbers"])
        md = save_summary_md(str(pdf), final["summary"],
                             final["numbers_total"], len(final["missing_numbers"]))
        rec = {"status": "done", "pdf": pdf.name, "md": md.name,
               "route": final.get("route", "?"),
               "conservation": f"{ok}/{final['numbers_total']}",
               "summary_head": _first_sentence(final.get("summary", "")),
               "seconds": round(time.time() - t0, 1)}
        _save_state(pdf, **rec)          # 先落 md，再写 done
        with _PRINT_LOCK:
            print(f"[完成] {tag}  守恒 {rec['conservation']}  路由 {rec['route']}"
                  f"  {rec['seconds']}s -> {md.name}")
        return rec
    except Exception as e:               # 单篇失败不拖累其他篇
        rec = {"status": "failed", "pdf": pdf.name,
               "error": f"{type(e).__name__}: {e}",
               "seconds": round(time.time() - t0, 1)}
        _save_state(pdf, **rec)
        with _PRINT_LOCK:
            print(f"[失败] {tag}  {rec['error']}")
        return rec

def _first_sentence(summary: str, limit: int = 90) -> str:
    """取总结第一句话当主线摘句（报告用，不额外调 LLM）。"""
    s = " ".join(summary.strip().split())
    cut = s.find("。")
    if 0 < cut + 1 <= limit:
        return s[:cut + 1]
    return s[:limit] + ("…" if len(s) > limit else "")

# ---------------- 汇总报告 ----------------
def write_report(results: list) -> pathlib.Path:
    n_done = sum(1 for r in results if r.get("status") == "done")
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    rows = sorted(results, key=lambda r: {"done": 0, "failed": 1}.get(r.get("status"), 2))

    lines = [
        "# 批量总结报告",
        "",
        f"- 论文数：{len(results)}｜完成 {n_done}（含跳过 {n_skip}）｜失败 {n_fail}",
        "",
        "| 论文 | 状态 | 数字守恒 | 路由 | 耗时(s) | 总结 md |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("status") == "done":
            status = "⏭ 已完成（跳过）" if r.get("skipped") else "✅"
        elif r.get("status") == "failed":
            status = f"❌ {r.get('error', '')[:60]}"
        else:
            status = str(r.get("status", "?"))
        lines.append(f"| {r.get('pdf', '?')} | {status} | {r.get('conservation', '—')} "
                     f"| {r.get('route', '—')} | {r.get('seconds', '—')} | {r.get('md', '—')} |")

    lines += ["", "## 各篇主线摘句", ""]
    for r in rows:
        if r.get("status") == "done":
            lines.append(f"- **{r['pdf']}**：{r.get('summary_head', '（无）')}")
        elif r.get("status") == "failed":
            lines.append(f"- **{r['pdf']}**：（失败）{r.get('error', '')}")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return REPORT_PATH

def main(pdf_dir: str | None = None) -> None:
    if "DEEPSEEK_API_KEY" not in os.environ:
        raise SystemExit("未检测到 DEEPSEEK_API_KEY。PowerShell 里先执行：\n"
                         '  $env:DEEPSEEK_API_KEY="sk-你的key"\n再重跑本脚本。')
    d = pathlib.Path(pdf_dir) if pdf_dir else BATCH_DIR
    pdfs = sorted(d.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"{d} 下没有找到 PDF")

    STATE_DIR.mkdir(exist_ok=True)
    print(f"批量总结 {len(pdfs)} 篇 | 目录 {d}\n"
          f"篇级并发 {BATCH_CONCURRENCY} ｜ map 并发全局 4 ｜ 断点续跑 .batch_state/\n")

    t0 = time.time()
    results: list = [None] * len(pdfs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_CONCURRENCY) as ex:
        futs = {ex.submit(run_one, p, i + 1, len(pdfs)): i
                for i, p in enumerate(pdfs)}
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut]] = fut.result()      # 按原顺序收结果

    report = write_report(results)
    n_done = sum(1 for r in results if r.get("status") == "done")
    print(f"\n全部结束：完成 {n_done}/{len(results)}，"
          f"总耗时 {round(time.time() - t0, 1)}s")
    print(f"报告已写入 {report.name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)