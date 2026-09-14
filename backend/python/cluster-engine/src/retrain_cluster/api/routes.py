"""HTTP 路由定义。

设计原则：路由层只做"准入校验 + 委托"，不承载业务逻辑。
所有错误一律通过抛出 ClusterError 交给统一处理器转换成 JSON。
"""

from dataclasses import asdict
from fastapi import APIRouter, Request, Response
from ..clustering.registry import API_ALGORITHMS, WARNINGS, available, validate_params
from ..config import model_fingerprint
from ..embeddings.registry import check_model
from ..retrieval.chroma import read_manifest
from ..errors import ClusterError
from .schemas import ClusteringRequest, ClusteringResponse, ErrorResponse

router = APIRouter()


@router.get("/health/live")
async def live():
    """存活探针：只要进程还在就返回 200（供容器编排判断是否重启）。"""
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(request: Request):
    """就绪探针：执行器已预热才返回 200，否则 503（供负载均衡摘除流量）。"""
    if not request.app.state.executor.ready():
        raise ClusterError("WORKER_NOT_READY", "Execution worker is warming up", 503)
    return {"status": "ready"}


@router.get("/api/v1/algorithms")
def algorithms(request: Request):
    """列出算法及其在当前环境的可用性（依赖是否已安装）。"""
    # max_samples 取服务级配置，而非写死常量，避免与 app.toml 脱节
    limit = request.app.state.settings.max_samples
    return {
        "algorithms": [
            {
                "algorithm": name,
                "available": available(name),
                "backend": "python",
                "implementation_version": "legacy-v1",
                "max_samples": limit,
                "warnings": WARNINGS.get(name, []),
            }
            for name in API_ALGORITHMS
        ]
    }


def profile_status(catalog, profile):
    """探测某个配置档当前是否可执行，返回 (是否可用, 不可用原因)。

    逐层检查：算法参数合法 -> 模型可用 -> （如启用检索）知识库完整。
    这里刻意"只查不建"，避免列表接口产生副作用。
    """
    try:
        validate_params(profile.algorithm, profile.algorithm_params, profile.backend)
        model = catalog.model(profile.model_id)
        check_model(model)
        if profile.features.n_results:
            read_manifest(
                catalog.settings.artifacts_dir / "knowledge_bases",
                profile.knowledge_base_id,
                model_fingerprint(model),
                model["dimension"],
            )
        return True, None
    except ClusterError as exc:
        return False, exc.code


def purification_status(profile):
    """净化依赖的可用性快照（无副作用，不加载权重）。

    净化是**软依赖**：本地 LLM 缺失时会自动降级为规则净化，因此它不参与
    ``available`` 判定（那会让 profile 在缺模型时整体不可用）。但降级状态必须
    被如实下发，调用方据此决定是否接受"这次净化其实是正则做的"。
    """

    spec = getattr(profile, "purification", None)
    if spec is None:
        return None
    from ..purification import probe

    return probe(spec).as_dict()


@router.get("/api/v1/profiles")
def profiles(request: Request):
    """列出可通过 API 使用的配置档及其可用状态。"""
    catalog = request.app.state.catalog
    result = []
    for profile in catalog.profiles.values():
        # 只暴露 API 白名单内的算法（mean_shift 等仅限 CLI）
        if profile.algorithm not in API_ALGORITHMS:
            continue
        ok, reason = profile_status(catalog, profile)
        purification = purification_status(profile)
        warnings = list(WARNINGS.get(profile.algorithm, []))
        if purification and purification.get("degraded"):
            # 降级不是错误，但必须让调用方看见
            warnings.append("PURIFIER_DEGRADED_TO_RULE")
        # 不对外发布文件系统路径与 provider 原始配置
        result.append(
            {
                "profile_id": profile.profile_id,
                "algorithm": profile.algorithm,
                "model_id": profile.model_id,
                "knowledge_base_id": profile.knowledge_base_id,
                "features": asdict(profile.features),
                "algorithm_params": profile.algorithm_params,
                "compatibility_status": profile.compatibility_status,
                "implementation_version": profile.implementation_version,
                "max_samples": min(catalog.settings.max_samples, profile.max_samples),
                "available": ok,
                "unavailable_reason": reason,
                "warnings": warnings,
                "purification": purification,
            }
        )
    return {"profiles": result}


@router.post(
    "/api/v1/clusterings",
    status_code=201,
    response_model=ClusteringResponse,
    responses={code: {"model": ErrorResponse} for code in [404, 413, 422, 429, 500, 502, 503, 504]},
)
def cluster(body: ClusteringRequest, request: Request, response: Response):
    """提交一次聚类请求。

    这里是"快速失败"的准入关口：配置档、算法白名单、样本数、文本长度逐一检查，
    全部通过后才交给执行器（因为执行器里可能发生计费的嵌入调用）。
    """
    state = request.app.state
    profile = state.catalog.profile(body.profile_id)
    if profile.algorithm not in API_ALGORITHMS:
        raise ClusterError("ALGORITHM_NOT_EXPOSED", "Algorithm is CLI-only", 422)
    if len(body.items) > min(state.settings.max_samples, profile.max_samples):
        raise ClusterError("INVALID_SAMPLE_COUNT", "Sample count exceeds this profile's limit", 422)
    if any(len(item.text) > state.settings.max_text_length for item in body.items):
        raise ClusterError("INVALID_TEXT", "Text exceeds configured length limit", 422)
    result = state.executor.execute(body.model_dump())
    # 按 REST 惯例用 Location 头指向新建结果的查询地址
    response.headers["Location"] = "/api/v1/clusterings/" + result["run_id"]
    return result


@router.get("/api/v1/clusterings/{run_id}", response_model=ClusteringResponse)
def get_result(run_id: str, request: Request):
    """按 run_id 查询历史结果（读取时会做完整性校验）。"""
    return request.app.state.runs.get(run_id)
