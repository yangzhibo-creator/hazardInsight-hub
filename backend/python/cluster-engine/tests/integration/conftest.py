"""集成测试共用清理：每个用例结束后释放显存。

为什么必须显式做：Qwen3.8-27B 常驻约 75–82 GiB 且**只能装一份**（陷阱 P3）。
同一个 pytest 进程里，前一个用例的模型对象即使已经离开作用域，PyTorch 的显存
缓存也不会立刻归还；后一个用例再加载就会：
``torch.OutOfMemoryError: Tried to allocate 48.91 GiB``。
上游没有跨用例的清理钩子，因此这里在每个 integration 用例之后统一
``gc.collect()`` + ``torch.cuda.empty_cache()``，让"一次只装一份"在测试进程里也成立。
"""

from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def release_gpu_memory_after_each_test():
    yield
    gc.collect()
    try:  # pragma: no cover - 无 GPU/无 torch 的环境直接跳过
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001 - 清理失败不应把用例判为失败
        pass
    gc.collect()
