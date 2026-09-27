from parser import DoclingParser
from collections import Counter

import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

def estimate_tokens(text: str) -> int:
    """粗估 token: 中文 ~1.5 字/token, 其余 ~4 字符/token"""
    cjk = len(_CJK.findall(text))
    return int(cjk / 1.5 + (len(text) - cjk) / 4) + 1

class MapReduceChunker:
    def __init__(self, max_tokens: int = 3000, min_tokens: int = 400):
        # DeepSeek 便宜、上下文大：块给大点，map 调用少、reduce 丢失少
        # 超过 ~4000 单块摘要质量开始下降
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk(self, blocks: list[dict]) -> list[dict]:
            chunks: list[dict] = []
            buf: list[dict] = []          # 当前攒着的 blocks
            buf_tokens = 0

            def flush():
                nonlocal buf, buf_tokens
                if buf:
                    chunks.append(self._pack(buf)) # 分块
                    buf, buf_tokens = [], 0

            for blk in blocks:
                # level<=2 的标题开新块
                if blk["type"] == "heading" and (blk.get("level") or 1) <= 2:
                    flush()
                    buf.append(blk)
                    continue

                buf.append(blk) # 其余 正文、公式、表格、代码、小标题
                buf_tokens += estimate_tokens(blk["text"])

                # 攒超了就在 block 边界落地（原子块在里面，天然不会被拦腰切）
                if buf_tokens > self.max_tokens:
                    flush()

            flush()
            return self._merge_tiny(chunks)

    def _pack(self, blocks: list[dict]) -> dict:
        parts: list[str] = []
        for b in blocks:
            t = (b.get("text") or "").strip()
            if not t:
                continue
            if b["type"] == "heading":
                lvl = min(int(b.get("level") or 1), 6)
                parts.append("#" * lvl + " " + t)   # 标题还原成 markdown，LLM 能看懂层级
            else:
                parts.append(t)

        text = "\n\n".join(parts)
        section = next((b.get("section") for b in blocks if b.get("section")), "")

        return {
            "text": text,
            "section": section,
            "types": [b["type"] for b in blocks],
            "tokens": estimate_tokens(text),
            "block_ids": [b["idx"] for b in blocks],
            "pages": sorted({b.get("page") for b in blocks if b.get("page")}),
            "atomic": len(blocks) == 1 and blocks[0]["type"] in {"table", "formula", "code"},
        }

    def _merge_tiny(self, chunks: list[dict]) -> list[dict]:
            """过短的纯正文块并回前一块, 原子块不参与合并"""
            merged: list[dict] = []
            for c in chunks:
                prev = merged[-1] if merged else None
                if (prev is not None and not c["atomic"] and not prev["atomic"]
                        and c["tokens"] < self.min_tokens
                        and prev["tokens"] + c["tokens"] <= self.max_tokens):
                    merged[-1] = {**prev,
                                  "text": prev["text"] + "\n\n" + c["text"],
                                  "types": prev["types"] + c["types"],
                                  "tokens": prev["tokens"] + c["tokens"],
                                  "block_ids": prev["block_ids"] + c["block_ids"],
                                  "pages": sorted({*prev["pages"], *c["pages"]})}
                else:
                    merged.append(c)
    
            for i, c in enumerate(merged):
                c["chunk_id"] = i
            return merged

if __name__ == "__main__":
    PDF = r"D:\githubProject\document summarize  agent\test doc\Clustering_based_Autoencoder_for_Anomaly_Detection1.pdf"
        
    parser = DoclingParser()
    data = parser.load_or_parse(PDF)   # 命中缓存，秒回
    blocks = data["blocks"]                 # 只需这一个字段
    chunks = MapReduceChunker().chunk(blocks)   # 分块

    total_tokens = sum(c["tokens"] for c in chunks)
    print(f"blocks: {len(blocks)} | chunks: {len(chunks)} | 合计 ~{total_tokens} tok")
    print("chunk 类型构成:", Counter(t for c in chunks for t in c["types"]))

    # 抽样预览前两块 + 所有含表格的块，肉眼确认内容完整可读
    preview = chunks[:2] + [c for c in chunks if "table" in c["types"]]
    for c in preview:
        print("\n" + "-" * 72)
        print(f"[#{c['chunk_id']} | {c['tokens']}t | {c['section']}]")
        print(c["text"])