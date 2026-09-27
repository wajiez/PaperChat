from __future__ import annotations

import hashlib
import json
import os
import pathlib

from openai import OpenAI

from summarizer import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

CHAT_DIR = pathlib.Path(__file__).parent / ".chat_sessions"
CHAT_TEMPERATURE = 0.2
MAX_HISTORY_ROUNDS = 10   # 发给模型的历史窗口

CHAT_SYSTEM_PROMPT = (
    "你是论文问答助手。只依据给定的论文材料回答用户问题；"
    "材料中没有的信息就明确说\"材料中没有\"，禁止编造或补充外部知识。"
    "数字必须与材料一致，并注明归属（谁的什么指标、在什么条件下）"
    "用纯文本书写数学符号（如 λ1），不要用 LaTeX 标记。"
)

def pdf_sha1(pdf_path: str) -> str:
    """PDF 内容哈希：同一篇论文的会话存取键"""
    return hashlib.sha1(pathlib.Path(pdf_path).read_bytes()).hexdigest()[:16]

def build_materials(blocks: list[dict], max_chars: int = 100000) -> str:
    """parse 缓存的 blocks 拼成问答材料"""
    parts = []
    for b in blocks:
        t = (b.get("text") or "").strip()
        if not t:
            continue
        sec = b.get("section") or ""
        if sec and b.get("type") != "heading":
            parts.append(f"[{sec}] {t}")
        else:
            parts.append(t)
    return "\n\n".join(parts)[:max_chars]

class PaperChat:
    """单篇论文的追问会话。messages 只存问答对（不含 system），
    system 携带全文材料，轮数增加上下文不膨胀。"""

    def __init__(self, client: OpenAI, materials_text: str,
                 summary: str = "", messages: list[dict] | None = None):
        self.client = client
        self.model = DEEPSEEK_MODEL
        self.summary = summary
        self._system = {"role": "system",
                        "content": CHAT_SYSTEM_PROMPT + "\n\n【论文全文材料】\n" + materials_text}
        if messages is not None:
            self.messages = list(messages)
        else:
            # 总结作为 assistant 首条：模型知道自己"说过什么"，
            # 用户问"你上面提到的 X"能对上
            self.messages = ([{"role": "assistant", "content": summary}]
                             if summary else [])
    def _payload(self) -> list[dict]:
            """发送视图：system + 总结（首条）+ 最近 N 轮完整问答对。
            self.messages 里的全量历史不动，裁剪只发生在发给 API 之前。"""
            head = self.messages[:1]                       # 总结那一条永远带上
            tail = self.messages[1:]
            keep = min(len(tail) // 2 * 2, MAX_HISTORY_ROUNDS * 2)   # 按完整轮对齐
            return [self._system] + head + tail[-keep:]
    
    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._payload(),
            temperature=CHAT_TEMPERATURE,
            max_tokens=1500,
        )
        answer = resp.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": answer})
        return answer

# ---------------- 会话持久化 ----------------
def _session_path(pdf_path: str) -> pathlib.Path:
    CHAT_DIR.mkdir(exist_ok=True)
    return CHAT_DIR / f"{pdf_sha1(pdf_path)}.json"


def save_session(chat: PaperChat, pdf_path: str) -> None:
    _session_path(pdf_path).write_text(json.dumps(
        {"summary": chat.summary, "messages": chat.messages},
        ensure_ascii=False, indent=1), encoding="utf-8")


def load_session(pdf_path: str) -> dict | None:
    f = _session_path(pdf_path)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None

def save_summary_md(pdf_path: str, summary: str,
                    numbers_total: int = 0, missing: int = 0) -> pathlib.Path:
    """最终总结落成 md 文件（按论文名区分，批量多篇不互相覆盖），
    返回文件路径供 main 打印给用户。只在首次跑完管线时写一次。"""
    stem = pathlib.Path(pdf_path).stem
    out = pathlib.Path(__file__).parent / f"final_summary_{stem}.md"
    body = summary if summary.lstrip().startswith("#") else f"# {stem} — 最终总结\n\n{summary}" # 确保 summary 文本以 Markdown 一级标题（#）开头，如果不是，就自动加上一个标题
    out.write_text(body + "\n", encoding="utf-8-sig")
    return out

# ---------------- 测试：总结管线 + 交互追问 ----------------
if __name__ == "__main__":
    from parser import DoclingParser

    PDF = r"D:\githubProject\document summarize  agent\test doc\Clustering_based_Autoencoder_for_Anomaly_Detection1.pdf"

    data = DoclingParser().load_or_parse(PDF)          # 解析缓存命中，秒回
    if "DEEPSEEK_API_KEY" not in os.environ:
        raise SystemExit("未检测到 DEEPSEEK_API_KEY。PowerShell 里先执行：\n"
                         '  $env:DEEPSEEK_API_KEY="sk-你的key"\n再重跑本脚本。')
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=DEEPSEEK_BASE_URL)

    saved = load_session(PDF)
    if saved:
        chat = PaperChat(client, build_materials(data["blocks"]),
                         summary=saved.get("summary", ""),
                         messages=saved["messages"])
        print(f"已恢复会话（{len(chat.messages) // 2} 轮历史，总结复用，零成本）")
    else:
        print("无历史会话，先跑总结管线（LangGraph 图，map 走摘要缓存）...")
        from graph import build_graph        # 局部导入避免循环依赖
        app = build_graph(client)
        final = app.invoke({"pdf_path": PDF})["final"]
        ok = final["numbers_total"] - len(final["missing_numbers"])
        print(f"总结完成（路由: {final['route']}），数字守恒 {ok}/{final['numbers_total']}\n")
        chat = PaperChat(client, build_materials(data["blocks"]),
                         summary=final["summary"])
        save_session(chat, PDF)
        md_path = save_summary_md(PDF, final["summary"],
                                  final["numbers_total"],
                                  len(final["missing_numbers"]))
        print(f"总结已写入：{md_path}")

    print("=" * 72)
    print("论文追问模式（输入 q 退出；问题只依据论文材料回答）")
    print("=" * 72)
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            break
        print("\n助手>", chat.ask(q))
        save_session(chat, PDF)      # 每轮落盘，中途退出也不丢
    print("(会话已保存，下次进入可继续追问)")