"""构建知识库（离线一次性操作）。

流程：写"构建中"清单 -> 编码全部文本 -> 建集合并分批写入 -> 校验记录数 ->
补写内容校验和并标记完成。清单状态从 building 到 completed，读取方只会接受后者。

除历史 CLI 的「原文逐行入库」（``processing="raw-lines-v1"``）外，这里也承载
「偏差数据库」页面上传语料后由 LLM 净化生成的语义知识库，因此额外支持：

* ``values``：调用方已经算好的向量（避免再编码一次，也便于分块上报进度）；
* ``entries.jsonl``：把真正写入索引的文本落一份旁路快照——ChromaDB 里只存向量，
  没有它，"查看知识库"就只能看到条目数，看不到内容；
* ``processing`` / ``display_name`` / ``source_name`` / ``created_at``：写进清单，
  让列表与详情能如实说明这个库"是怎么来的"。
"""

from importlib.metadata import version
import json
import uuid
from .chroma import kb_path, collection_checksum
from ..artifacts.runs import atomic_json
from ..artifacts.fingerprints import fingerprint, file_hash
from ..data.validation import validate_matrix

#: 旁路文本快照的文件名（每行一条 JSON，便于流式读取与分页）。
ENTRIES_FILE = "entries.jsonl"
#: 原始语料快照的文件名（可选；上传生成的库会写一份，便于溯源）。
CORPUS_FILE = "corpus.txt"


def _write_entries(path, texts) -> None:
    """把写入索引的文本逐行落盘（UTF-8、不转义中文）。"""

    with (path / ENTRIES_FILE).open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _write_corpus(path, corpus_texts) -> None:
    """把上传的原始语料落一份快照；清洗后的文本另存 entries.jsonl。"""

    with (path / CORPUS_FILE).open("w", encoding="utf-8") as handle:
        for text in corpus_texts:
            handle.write(str(text).replace("\n", " ") + "\n")


def build_knowledge_base(
    directory,
    ident,
    texts,
    encoder,
    model_fingerprint,
    dimension,
    source_path=None,
    *,
    values=None,
    processing="raw-lines-v1",
    display_name=None,
    source_name=None,
    source_sha256=None,
    corpus_texts=None,
    created_at=None,
    extra_meta=None,
):
    """构建并落盘一个知识库。

    参数:
        directory: 知识库根目录（``artifacts/knowledge_bases``）。
        ident: 知识库标识；必须匹配白名单正则（见 :func:`kb_path`）。
        texts: 真正写入索引的文本（净化后或原文，取决于 ``processing``）。
        encoder: 文本编码器；``values`` 已给出时不会被调用（可为 None）。
        model_fingerprint: 编码模型指纹，检索端会逐字比对。
        dimension: 向量维度，检索端会逐字比对。
        source_path: 来源文件路径；给出时用于计算 ``source_sha256``（历史 CLI 口径）。
        values: 预先算好的向量矩阵；与 ``texts`` 行数一致时优先使用。
        processing: 预处理口径标识（如 ``raw-lines-v1`` / ``purified-qwen-v1``）。
        display_name / source_name / created_at: 写进清单的可读元信息。
        source_sha256: 未给 ``source_path`` 时可直接指定来源内容指纹。
        corpus_texts: 上传的原始语料；给出时额外落一份 ``corpus.txt`` 快照。
        extra_meta: 追加到清单的其它键（调用方自定义），不得覆盖保留字段。
    """

    path = kb_path(directory, ident)
    path.mkdir(parents=True, exist_ok=False)  # 绝不覆盖已存在的索引，避免半新半旧
    meta = {
        "schema_version": 1,
        "knowledge_base_id": ident,
        "status": "building",  # 先落一个"未完成"状态，中断时不会被误用
        "verification": "verified",
        "processing": processing,
        "metric": "l2",  # 距离度量；与检索端的校验必须一致
        "model_fingerprint": model_fingerprint,
        "dimension": dimension,
        "count": len(texts),
        "source_sha256": file_hash(source_path) if source_path is not None else (source_sha256 or ""),
        "texts_fingerprint": fingerprint(texts),
        "chromadb_version": version("chromadb"),  # 记录版本，便于排查索引兼容问题
        "index_origin": "new-current-environment",
    }
    if display_name:
        meta["display_name"] = str(display_name)
    if source_name:
        meta["source_name"] = str(source_name)
    if created_at:
        meta["created_at"] = str(created_at)
    if extra_meta:
        for key, value in extra_meta.items():
            if key in meta:
                raise ValueError(f"extra_meta must not override reserved keys: {key}")
            meta[key] = value
    atomic_json(path / "manifest.json", meta)
    if values is None:
        values = encoder.encode(texts)
    values = validate_matrix(values, rows=len(texts), dimension=dimension)
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(str(path / "index"), settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        name="micro", embedding_function=None, configuration={"hnsw": {"space": "l2"}}
    )
    # ID 用随机 UUID：知识库检索只需向量相似度，ID 本身无业务含义
    ids = [str(uuid.uuid4()) for _ in texts]
    # 分批写入，避免一次性提交导致内存峰值过高
    for start in range(0, len(texts), 2048):
        collection.add(ids=ids[start : start + 2048], embeddings=values[start : start + 2048])
    if collection.count() != len(texts):
        raise RuntimeError("Incomplete knowledge base insertion")
    # 旁路文本快照：必须在标记 completed 之前写好，否则"完成"的库里看不到内容
    _write_entries(path, texts)
    if corpus_texts is not None:
        _write_corpus(path, corpus_texts)
    # 全部写入成功后才记录校验和并置为 completed——这是"索引可用"的唯一凭证
    meta.update(status="completed", records_sha256=collection_checksum(collection))
    atomic_json(path / "manifest.json", meta)
    return meta
