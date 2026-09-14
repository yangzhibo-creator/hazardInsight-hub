"""命令行入口。

各子命令统一遵循"把结果 JSON 打到 stdout、错误打到 stderr 并返回非零退出码"的约定，
因此可安全地用于脚本与流水线。
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from .config import Settings, Catalog, model_fingerprint
from .artifacts.fingerprints import jsonable, file_hash
from .artifacts.runs import atomic_json
from .errors import ClusterError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="retrain-cluster")
    parser.add_argument("--config", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    # profiles：列出所有可用配置档
    commands.add_parser("profiles")
    # serve：启动 HTTP 服务
    serve = commands.add_parser("serve")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    # cluster：按配置档对一批文本做聚类
    cluster = commands.add_parser("cluster")
    cluster.add_argument("--input", required=True, help="JSON request containing profile_id and items")
    cluster.add_argument("--output")
    # build-kb：离线构建检索知识库
    build = commands.add_parser("build-kb")
    build.add_argument("--model-id", required=True)
    build.add_argument("--id", required=True)
    build.add_argument("--input", required=True)
    # tune：跑一次 Optuna 超参搜索实验
    tune = commands.add_parser("tune")
    tune.add_argument("--profile", required=True)
    tune.add_argument("--n-results", type=int, default=-1)
    tune.add_argument("--pca-dim", type=int, default=0)
    tune.add_argument("--n-trials", type=int)
    tune.add_argument("--n-jobs", type=int)
    tune.add_argument("--seed", type=int)
    # replay：用离线保存的向量复现历史结果（不调用编码服务）
    replay = commands.add_parser("replay")
    replay.add_argument("--profile", required=True)
    replay.add_argument("--embeddings", required=True)
    replay.add_argument("--labels")
    replay.add_argument("--output", required=True)
    # import-cache：导入历史实验遗留的向量
    imp = commands.add_parser("import-cache")
    imp.add_argument("--model-id", required=True)
    imp.add_argument("--input", required=True, help="JSON string list matching original embedding order")
    imp.add_argument("--legacy", required=True)
    # model-manifest：为本地模型目录生成校验清单
    inspect = commands.add_parser("model-manifest")
    inspect.add_argument("--directory", required=True)
    inspect.add_argument("--output", required=True)
    # download-model：从 HuggingFace 拉取指定 revision 的模型
    download = commands.add_parser("download-model")
    download.add_argument("--source", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
        # 前两个命令不依赖配置，可独立运行
        if args.command == "model-manifest":
            root = Path(args.directory).resolve()
            output = Path(args.output).resolve()
            # 列出目录下所有文件的相对路径与哈希；排除输出文件自身以免自我引用
            files = {
                str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.resolve() != output
            }
            if not files:
                raise ValueError("Model directory is empty")
            result = {"required_files": list(files), "file_checksums": files}
            atomic_json(output, result)
        elif args.command == "download-model":
            from huggingface_hub import snapshot_download

            # 禁止覆盖已存在的目录：避免把"部分下载"和"完整模型"混在一起
            if Path(args.destination).exists():
                raise ValueError("Download destination must not already exist")
            result = {
                "path": snapshot_download(repo_id=args.source, revision=args.revision, local_dir=args.destination)
            }
        else:
            settings = Settings.load(args.config)
            if args.command == "serve":
                import uvicorn
                from .api.app import create_app

                uvicorn.run(
                    create_app(settings=settings), host=args.host or settings.host, port=args.port or settings.port
                )
                return 0
            if args.command == "profiles":
                result = [asdict(p) for p in Catalog(settings).profiles.values()]
            elif args.command == "cluster":
                from .services.clustering import ClusteringService

                result = ClusteringService(settings).cluster(json.loads(Path(args.input).read_text(encoding="utf-8")))
                if args.output:
                    atomic_json(args.output, result)
            elif args.command == "build-kb":
                from .services.clustering import ClusteringService
                from .data.loaders import load_knowledge_texts
                from .retrieval.build import build_knowledge_base

                service = ClusteringService(settings)
                model = service.catalog.model(args.model_id)
                model_hash = model_fingerprint(model)
                result = build_knowledge_base(
                    settings.artifacts_dir / "knowledge_bases",
                    args.id,
                    load_knowledge_texts(args.input),
                    service.encoders.get(model, model_hash),
                    model_hash,
                    model["dimension"],
                    args.input,
                    # 让 CLI 构建的库与页面构建的库在"偏差数据库"列表里有同样的溯源信息
                    display_name=args.id,
                    source_name=Path(args.input).name,
                    model_id=args.model_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            elif args.command == "tune":
                from .services.experiments import ExperimentService

                result = ExperimentService(settings).run(
                    args.profile,
                    n_results=args.n_results,
                    pca_dim=args.pca_dim,
                    n_trials=args.n_trials,
                    n_jobs=args.n_jobs,
                    seed=args.seed,
                )
            elif args.command == "import-cache":
                from .artifacts.cache import EmbeddingCache

                catalog = Catalog(settings)
                texts = json.loads(Path(args.input).read_text(encoding="utf-8"))
                cache = EmbeddingCache(settings.artifacts_dir / "embeddings")
                key = cache.key(texts, model_fingerprint(catalog.model(args.model_id)))
                # 导入的向量标记为 unverified，读取时需显式放行
                cache.import_legacy(key, args.legacy)
                result = {"key": key, "verification": "unverified"}
            elif args.command == "replay":
                import numpy as np
                from .clustering.registry import validate_params
                from .features.pipeline import FeaturePipeline
                from .services.clustering import ClusteringService
                from .evaluation.metrics import QbEvaluator

                service = ClusteringService(settings)
                profile = service.catalog.profile(args.profile)
                model_hash = model_fingerprint(service.catalog.model(profile.model_id))
                # 直接用本地 .npy 向量，不调用编码服务（离线复现）
                values = np.load(args.embeddings, allow_pickle=False)
                transformed = FeaturePipeline(service.retriever(profile, model_hash)).transform(
                    values, profile.features
                )
                labels = validate_params(profile.algorithm, profile.algorithm_params)(
                    transformed, **profile.algorithm_params
                )
                result = {"profile": asdict(profile), "labels": labels, "baseline_kind": "current-environment"}
                # 提供真实标签时顺带算出评估指标
                if args.labels:
                    result["metrics"] = QbEvaluator()(np.load(args.labels, allow_pickle=False), labels)
                atomic_json(args.output, result)
        print(json.dumps(jsonable(result), ensure_ascii=False, indent=2))
        return 0
    except ClusterError as exc:
        # 业务错误以 JSON 形式输出到 stderr，并返回退出码 1
        print(json.dumps(exc.payload("cli")), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
