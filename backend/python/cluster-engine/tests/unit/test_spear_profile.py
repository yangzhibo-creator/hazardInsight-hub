"""spear-v1 在配置层的注册、解析与可用性报告。

profile 体系是这个引擎的"配置即契约"入口：新增实现版本必须能在
``configs/profiles/`` 里被加载、被校验、被如实报告依赖缺失原因。
本模块钉住这几点，避免 SPEAR 变成"只有代码里存在、配置层进不来"的能力。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrain_cluster.api.routes import purification_status, profile_status
from retrain_cluster.config import Catalog, Settings
from retrain_cluster.errors import ClusterError

ROOT = Path(__file__).resolve().parents[2]


def _write_config(tmp_path: Path, profile: dict) -> Settings:
    (tmp_path / "profiles").mkdir(exist_ok=True)
    (tmp_path / "models.toml").write_text(
        """[[models]]
model_id = "fixture"
provider = "openai_compatible"
source = "fixture"
base_url = "http://127.0.0.1:9/v1"
revision = "fixture-v1"
dimension = 8
""",
        encoding="utf-8",
    )
    (tmp_path / "app.toml").write_text(
        '[app]\nmodels_file="models.toml"\nprofiles_dir="profiles"\nartifacts_dir="artifacts"\n', encoding="utf-8"
    )
    (tmp_path / "profiles" / "spear.json").write_text(json.dumps(profile), encoding="utf-8")
    return Settings.load(tmp_path / "app.toml")


def _spear_profile(**overrides) -> dict:
    profile = {
        "profile_id": "spear_purified",
        "algorithm": "agglomerative",
        "model_id": "fixture",
        "algorithm_params": {"distance_threshold": 0.5},
        "features": {"beta": 1.0, "n_results": 0, "pca_dim": 0, "version": "spear-v1"},
        "purification": {
            "enabled": True,
            "backend": "auto",
            "model_path": "models/Qwen3.8-27B",
            "cache_file": "data/purify_map.json",
            "guard": True,
        },
        "implementation_version": "spear-v1",
    }
    profile.update(overrides)
    return profile


def test_spear_profile_is_parsed_and_paths_are_absolutised(tmp_path):
    settings = _write_config(tmp_path, _spear_profile())
    catalog = Catalog(settings)
    profile = catalog.profile("spear_purified")
    assert profile.implementation_version == "spear-v1"
    assert profile.features.version == "spear-v1"
    assert profile.purification is not None
    # 相对路径必须按配置目录解析成绝对路径，运行期不依赖 cwd
    assert Path(profile.purification.model_path).is_absolute()
    assert Path(profile.purification.cache_file).is_absolute()
    assert profile.capability == {}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"purification": None}, "requires a purification spec"),
        ({"features": {"beta": 1.0, "n_results": 0, "pca_dim": 0, "version": "legacy-v1"}}, "spear feature spec"),
        ({"algorithm": "semantic_auto_kmeans"}, "does not support semantic algorithms"),
        (
            {"features": {"beta": 0.5, "n_results": 6, "pca_dim": 0, "version": "spear-v1"}, "knowledge_base_id": None},
            "needs a knowledge base",
        ),
        ({"purification": {"enabled": True, "backend": "auto", "bogus_field": 1}}, "Profile fields are invalid"),
    ],
)
def test_spear_profile_rejects_illegal_combinations(tmp_path, overrides, message):
    profile = _spear_profile(**overrides)
    with pytest.raises(ClusterError, match=message):
        Catalog(_write_config(tmp_path, profile))


def test_purification_availability_is_reported_as_a_soft_dependency(tmp_path):
    """缺 LLM 时 profile 仍可用，但必须给出降级状态与原因。"""

    settings = _write_config(tmp_path, _spear_profile())
    catalog = Catalog(settings)
    profile = catalog.profile("spear_purified")

    status = purification_status(profile)
    assert status["enabled"] is True
    assert status["degraded"] is True
    assert status["effective_backend"] == "rule"
    assert status["reason"]
    # 向量模型是远程 provider 且没有凭据 → profile 本身不可用（硬依赖，与净化无关）
    ok, reason = profile_status(catalog, profile)
    assert ok is False and reason == "MODEL_UNAVAILABLE"


def test_purification_status_is_none_for_legacy_profiles(tmp_path):
    legacy = _spear_profile()
    legacy.pop("purification")
    legacy["implementation_version"] = "legacy-v1"
    legacy["features"] = {"beta": 1.0, "n_results": 0, "pca_dim": 0, "version": "legacy-v1"}
    catalog = Catalog(_write_config(tmp_path, legacy))
    assert purification_status(catalog.profile("spear_purified")) is None


def test_shipped_profiles_load_and_declare_their_dependencies():
    """随仓库发布的真实 profile 必须能被 Catalog 加载，且依赖声明完整。"""

    settings = Settings.load(ROOT / "configs" / "app.toml")
    catalog = Catalog(settings)
    assert catalog.profile("spear_purified").features.n_results == 0
    retrieval = catalog.profile("spear_purified_retrieval")
    assert retrieval.features.n_results == 6
    assert retrieval.knowledge_base_id == "bge-large-raw-lines-v1"
    assert retrieval.purification is not None and retrieval.purification.enabled is True
    # 两个 profile 只在"是否检索增强"上不同，其余口径必须一致
    assert retrieval.purification.backend == catalog.profile("spear_purified").purification.backend
