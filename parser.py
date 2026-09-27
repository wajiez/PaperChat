from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.document import (
    CodeItem,
    DoclingDocument,
    FormulaItem,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)

import hashlib
import json
import pathlib

import docling

# 三类不可分割单元
ATOMIC_TYPES = {"table", "formula", "code"}

CACHE_DIR = pathlib.Path(__file__).parent / ".parse_cache"

class DoclingParser:
    """
    使用 Docling 解析 PDF 文档 parse → serialize
    输出结构化结果
    """

    def __init__(self, enrich_formula_code: bool = True):
        # PDF 流水线配置
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,              # 如果是电子PDF可改为 False 提速
            do_table_structure = enrich_formula_code,  # 表格结构识别
            do_formula_enrichment = enrich_formula_code, # 公式结构识别
            do_code_enrichment = enrich_formula_code, # 代码结构识别
            images_scale=1.0,         # 降低图片缩放比例，减少显存占用
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        self._table_ser: MarkdownDocSerializer | None = None 

        # 缓存配置版本：开关状态或 docling 版本一变，旧缓存自动失效
        self._cfg_version = f"v1_{int(enrich_formula_code)}_{docling.__version__}"

    def parse(self, file_path: str) -> dict:
        result = self.converter.convert(file_path)
        document = result.document

        self._table_ser = None  # 每个文件重建

        markdown_text = document.export_to_markdown() # 把整份文档序列化成 markdown
        blocks = self._extract_blocks(document)

        return {
            "file_path": file_path,
            "text": markdown_text,          # 原始 markdown
            "document": document,           # 原始 DoclingDocument
            "blocks": blocks,               # ★ 拍平后的块序列，带类型与 atomic 标记
            "tables": [b for b in blocks if b["type"] == "table"],
            "formulas": [b for b in blocks if b["type"] == "formula"],
            "code_blocks": [b for b in blocks if b["type"] == "code"],
            "unbreakable_units": [b for b in blocks if b["atomic"]],
        }

    def cache_key(self, pdf_path: str) -> str:
        """PDF 内容哈希 + 解析配置版本。内容变了或配置变了，缓存自动失效。"""
        h = hashlib.sha1(pathlib.Path(pdf_path).read_bytes()).hexdigest()[:16]
        return f"{h}_{self._cfg_version}"

    def load_or_parse(self, pdf_path: str, keep_document: bool = False) -> dict:
        """带缓存的解析入口。返回结构与 parse() 的 return 完全一致。

        - 主缓存 {key}.json：存除 document 外的全部字段（纯 JSON 数据）
        - keep_document=True 时额外存/取 {key}.doc.json（docling 自带的
          save_as_json / load_from_json），取回后拼回 dict，结构与 parse() 一致
        - keep_document=False（默认）：data["document"] 为 None，占住 key
        """
        key = self.cache_key(pdf_path)
        cache_file = CACHE_DIR / f"{key}.json"

        if cache_file.exists():  # 命中：秒回
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if keep_document:
                doc_file = CACHE_DIR / f"{key}.doc.json"
                if doc_file.exists():
                    data["document"] = DoclingDocument.load_from_json(doc_file)
            return data

        data = self.parse(pdf_path)  # 未命中
        doc = data.pop("document") 
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if keep_document:
            doc.save_as_json(CACHE_DIR / f"{key}.doc.json")

        data["document"] = doc  # 内存里这次调用照旧无损
        return data

    def _extract_blocks(self, document) -> list[dict]:
        """
        按阅读顺序遍历 docling 的结构树，把每个 item 变成带类型的 block。
        """
        blocks: list[dict] = []
        heading_stack: list[tuple[int, str]] = [] # 标题栈
        skip_refs: set[str] = set() # 去重跳过名单

        # 遍历 item
        for item, _level in document.iterate_items( 
            included_content_layers={ContentLayer.BODY}
        ):
            if item.self_ref in skip_refs:
                continue
            if isinstance(item, PictureItem):
                continue  # 不管图片（图注作为普通正文留下）

            page = item.prov[0].page_no if getattr(item, "prov", None) else None

            # Section
            if isinstance(item, SectionHeaderItem):
                while heading_stack and heading_stack[-1][0] >= item.level:
                    heading_stack.pop()
                heading_stack.append((item.level, item.text))
                blocks.append(
                    self._mk(
                        "heading", item.text, page, self._section(heading_stack),
                        atomic=False, level=item.level,
                    )
                )
                continue

            section = self._section(heading_stack)

            # 原子块：表格 / 公式 / 代码
            kind = self._atomic_kind(item)
            if kind:
                blocks.append(
                    self._mk(
                        kind,
                        self._render_atomic(item, kind, document),
                        page, section,
                        atomic=True,
                        meta=self._atomic_meta(item, kind, document),
                    )
                )
                skip_refs |= self._sub_refs(item)
                continue

            # Text
            if isinstance(item, (TextItem, ListItem)) and (item.text or "").strip():
                blocks.append(
                    self._mk(
                        "prose", item.text, page, section,
                        atomic=False, meta={"label": str(item.label)},
                    )
                )

        for i, b in enumerate(blocks): # (序号, 元素)
            b["idx"] = i # new key idx
        return blocks    

    @staticmethod
    def _mk(type_: str, text: str, page, section: str, atomic: bool,
            level: int | None = None, meta: dict | None = None) -> dict:
        return {
            "type": type_,
            "text": text,
            "page": page,
            "section": section,
            "atomic": atomic,
            "level": level, # 只有 heading 有
            "meta": meta or {}, # meta 只有原子块有
        }

    @staticmethod
    def _section(stack: list[tuple[int, str]]) -> str: #  把标题栈拼成一条路径
        return " > ".join(t for _, t in stack)

    @staticmethod
    def _atomic_kind(item) -> str | None:
        if isinstance(item, TableItem):
            return "table"
        if isinstance(item, FormulaItem):
            return "formula"
        if isinstance(item, CodeItem):
            return "code"
        return None

    @staticmethod
    def _sub_refs(item) -> set[str]:
        """表格的 rich cell 子项、caption 会被 iterate_items 再遍历一次，收集起来跳过。"""
        refs = {c.cref for c in (getattr(item, "children", None) or [])}
        refs |= {c.cref for c in (getattr(item, "captions", None) or [])}
        return refs


    def _render_atomic(self, item, kind: str, document) -> str: # text
        if kind == "table":
            if self._table_ser is None:
                self._table_ser = MarkdownDocSerializer(
                    doc=document,
                    params=MarkdownParams(compact_tables=True),
                )
            return self._table_ser.serialize(item=item).text.strip()

        if kind == "formula":
            # 统一成展示式公式，避免和正文里的行内 $...$ 混淆
            return f"$$\n{(item.text or '').strip()}\n$$"

        if kind == "code":
            lang = (getattr(item, "code_language", None) or "").strip()
            return f"```{lang}\n{(item.text or '').rstrip()}\n```"

        return ""

    def _atomic_meta(self, item, kind: str, document) -> dict:
        meta: dict = {}

        if kind == "table":
            data = item.data
            meta["num_rows"] = data.num_rows
            meta["num_cols"] = data.num_cols
            # 用 column_header 标记取表头
            headers = sorted(
                (c for c in data.table_cells if getattr(c, "column_header", False)),
                key=lambda c: (c.start_row_offset_idx, c.start_col_offset_idx),
            )
            if headers:
                meta["headers"] = [(c.text or "").strip() for c in headers]

        if kind == "code" and getattr(item, "code_language", None):
            meta["language"] = item.code_language

        # 表注 / 公式注（「Table 2: ...」这类信息量很大，单独留一份供下游拼进检索文本）
        caps: list[str] = []
        for ref in (getattr(item, "captions", None) or []):
            try:
                cap = ref.resolve(doc=document)
            except Exception:
                continue
            if (getattr(cap, "text", "") or "").strip():
                caps.append(cap.text.strip())
        if caps:
            meta["captions"] = caps

        return meta

    
if __name__ == "__main__":
    from collections import Counter

    PDF = r"D:\githubProject\document summarize  agent\test doc\Clustering_based_Autoencoder_for_Anomaly_Detection1.pdf"
    
    parser = DoclingParser()
    data = parser.load_or_parse(PDF) 

    print("markdown 长度:", len(data["text"]))
    blocks = data["blocks"]
    print("blocks 总数:", len(blocks))
    print("块类型分布:", dict(Counter(b["type"] for b in blocks)))

    atomic = data["unbreakable_units"]
    print(f"不可分割单元: {len(atomic)} 个 "
        f"({dict(Counter(b['type'] for b in atomic))})")
    
    # 缓存一致性抽查：取出的 blocks 应与解析结果逐块一致
    again = parser.load_or_parse(PDF)
    print("缓存取回一致:", again["blocks"] == data["blocks"])