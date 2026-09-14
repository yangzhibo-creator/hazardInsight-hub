"""Request/response contracts and route-level profile gating."""

import pytest
from pydantic import ValidationError
from retrain_cluster.api.routes import profile_status
from retrain_cluster.api.schemas import ClusteringRequest, ClusteringResponse, Item
from retrain_cluster.config import Catalog
from retrain_cluster.errors import ClusterError


def item(**kwargs):
    defaults = {"id": "a", "text": "content"}
    return Item(**{**defaults, **kwargs})


def request(**kwargs):
    defaults = {"profile_id": "fixture", "items": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]}
    return ClusteringRequest(**{**defaults, **kwargs})


def test_item_requires_non_empty_id_and_text():
    with pytest.raises(ValidationError):
        item(id="")
    with pytest.raises(ValidationError):
        item(text="   ")


def test_item_is_strict_about_types():
    with pytest.raises(ValidationError):
        item(id=123)
    with pytest.raises(ValidationError):
        item(text=None)


def test_item_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        item(extra="nope")


@pytest.mark.parametrize(
    "change",
    [
        {"items": [{"id": "a", "text": "x"}]},
        {"profile_id": ""},
        {"items": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]},
    ],
)
def test_request_enforces_minimum_size_and_unique_ids(change):
    with pytest.raises(ValidationError):
        request(**change)


def test_response_serialization_roundtrip():
    payload = {
        "run_id": "run_" + "0" * 32,
        "profile_id": "fixture",
        "algorithm": "agglomerative",
        "implementation_version": "legacy-v1",
        "n_samples": 2,
        "n_clusters": 1,
        "n_noise": 0,
        "assignments": [{"id": "a", "cluster_id": 0}],
        "warnings": [],
        "elapsed_ms": 3,
        # spear-v1 新增的可选字段：legacy 路径恒为 None，序列化后必须仍在
        "purification": None,
    }
    assert ClusteringResponse(**payload).model_dump() == payload


def test_profile_status_accepts_a_ready_profile(settings, monkeypatch):
    monkeypatch.setattr("retrain_cluster.api.routes.check_model", lambda spec: None)
    catalog = Catalog(settings)
    profile = catalog.profile("fixture")
    assert profile_status(catalog, profile) == (True, None)


def test_profile_status_reports_unavailable_models(settings, monkeypatch):
    def unavailable(spec):
        raise ClusterError("MODEL_UNAVAILABLE", "no credential", 503)

    monkeypatch.setattr("retrain_cluster.api.routes.check_model", unavailable)
    catalog = Catalog(settings)
    assert profile_status(catalog, catalog.profile("fixture")) == (False, "MODEL_UNAVAILABLE")


def test_profile_status_rejects_invalid_algorithm_parameters(settings, monkeypatch):
    monkeypatch.setattr("retrain_cluster.api.routes.check_model", lambda spec: None)
    catalog = Catalog(settings)
    profile = catalog.profile("fixture")
    broken = profile.__class__(**{**profile.__dict__, "algorithm_params": {"distance_threshold": -1}})
    ok, reason = profile_status(catalog, broken)
    assert ok is False and reason == "INVALID_PROFILE"
