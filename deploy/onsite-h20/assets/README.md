# 构建资产（不进 git 的大文件放这里）

`build_image.sh` 在构建镜像前把现场交付介质里的资产复制到本目录，Dockerfile
只从 `assets/` 读取。仓库本身**不包含**这些大文件。

需要的目录结构：

```
deploy/onsite-h20/assets/
├── models/
│   └── bge-large-zh-v1.5/          # 约 1.3 GB，进镜像（论文首选表示组件）
├── knowledge_bases/
│   └── bge-large-raw-lines-v1/     # 预置 ChromaDB 索引 + manifest.json
└── purify_map.json                 # 可选：预计算净化映射（原文 -> 浓缩文本）
```

## 知识库为什么必须是 `knowledge_bases/<id>/`

cluster-engine 的 `ChromaRetriever` 打开索引前会做三重一致性校验（记录数、
HNSW 距离空间、内容校验和），并要求同目录下的 `manifest.json` 里
`status=completed` / `verification=verified` / `model_fingerprint` / `dimension` 全部匹配。
少一样都会以 `KNOWLEDGE_BASE_UNAVAILABLE` 拒绝加载，而不是"凑合用"。

用打包机上的引擎生成（等价于历史 `run_all_algos.py` 的读法：保留 `database.txt`
每行开头的 `- ` 前缀）：

```bash
cd backend/python
python -m retrain_cluster build-kb \
  --model-id bge-large-zh-v1.5 \
  --id bge-large-raw-lines-v1 \
  --input ../data/database.txt
# 产物：cluster-engine/artifacts/knowledge_bases/bge-large-raw-lines-v1/{manifest.json,index/}
```

**HNSW 建索引带随机性**：现场数值要与验证过的数值一致，就必须分发同一份索引，
而不是现场重建（检索结果对索引构建顺序敏感）。

## Qwen3.8-27B 为什么不在这里

约 55 GB，打进镜像会让 tar 包过大、现场 `docker load` 挤占 70 分钟预算。
它由 `start_container.sh` 以只读卷挂载到
`/app/cluster-engine/models/Qwen3.8-27B`（与 profile 里的相对路径一致）。
