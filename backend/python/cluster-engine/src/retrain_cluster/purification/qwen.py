"""本地 Qwen3.8-27B 净化器（VL 架构，本任务只走文本通道）。

三条来自真机实测的约束（陷阱 P3/P4）：

* **常驻约 75 GiB 显存，只能装一份**：第二个加载 Qwen 的进程会直接 CUDA OOM，
  因此实例复用由 :mod:`registry` 负责，本类只做"加载一次、eval、批量生成"；
* **不要用 ``processor.apply_chat_template(..., return_dict=True)``**：
  transformers 5.x 下可能返回 list 而非张量，导致 ``generate`` 报
  ``'list' object has no attribute 'shape'``。这里直接走底层 ``tokenizer``；
* **批量生成必须左填充**，否则短文本的生成位置会对不齐。

另外显式关闭 thinking 模式以提速；chat template 不认该参数时用 ``except TypeError``
兼容，而不是让整个加载失败。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

from .base import BasePurifier
from .prompts import SYSTEM_PROMPT, build_prompt

log = logging.getLogger("retrain_cluster.purification.qwen")

#: 单条提示的 token 上限。净化输入是短文本，2048 足够容纳少样本示例。
MAX_PROMPT_TOKENS = 2048
#: 生成结果保留的字符上限，防御模型偶发复读。
MAX_OUTPUT_CHARS = 64


class QwenPurifier(BasePurifier):
    """本地 Qwen 净化器；权重缺失时构造阶段不报错，``load()`` 才报。"""

    name = "qwen"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        batch_size: int = 32,
        max_new_tokens: int = 64,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ 加载
    def load(self) -> "QwenPurifier":
        """加载权重与 processor；只允许调用一次（实例复用见 registry）。"""

        if self._model is not None:
            return self
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"purification model directory is missing: {self.model_path}")

        import torch
        from transformers import AutoProcessor

        log.info("loading purification model: %s (device=%s)", self.model_path, self.device)
        self._processor = AutoProcessor.from_pretrained(str(self.model_path), trust_remote_code=True)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            # 生成式模型批量推理必须左填充
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        model_class = self._resolve_model_class()
        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self._model = model_class.from_pretrained(
            str(self.model_path),
            dtype=dtype,
            device_map=self.device if str(self.device).startswith("cuda") else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        log.info("purification model ready: %s", type(self._model).__name__)
        return self

    @staticmethod
    def _resolve_model_class():
        """按 transformers 版本选择可加载 qwen3_5 架构的模型类。

        不同 transformers 版本暴露的类名不同，逐个探测而不是硬编码一个，
        这样基础镜像换版本时不会因为"类不存在"这种无关原因失败。
        """

        import transformers

        for attribute in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen3_5ForConditionalGeneration",
        ):
            candidate = getattr(transformers, attribute, None)
            if candidate is not None:
                return candidate
        raise RuntimeError(
            "installed transformers does not expose a qwen3_5-capable model class; "
            "upgrade transformers to a version that recognises the qwen3_5 architecture"
        )

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        """底层 tokenizer；processor 只管 VL 的文本通道时直接用它。"""

        return getattr(self._processor, "tokenizer", None) or self._processor

    # ------------------------------------------------------------------ 推理
    def _build_inputs(self, prompts: list[str]):
        import torch

        tokenizer = self.tokenizer
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        texts = []
        for message in messages:
            try:
                texts.append(
                    tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                )
            except TypeError:
                # 该 chat template 不认 enable_thinking 参数
                texts.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            add_special_tokens=False,
        )
        if not hasattr(encoded, "keys"):
            raise RuntimeError(f"tokenizer returned an unexpected type: {type(encoded)}")
        return {key: value.to(self.model.device) for key, value in dict(encoded).items() if value is not None}

    def purify(self, texts: Sequence[str]) -> list[str]:
        sources = [str(text) for text in texts]
        if not sources:
            return []
        import torch

        outputs: list[str] = []
        for start in range(0, len(sources), self.batch_size):
            chunk = sources[start : start + self.batch_size]
            prompts = [build_prompt(text) for text in chunk]
            inputs = self._build_inputs(prompts)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # 确定性输出，便于现场复现
                    num_beams=1,
                )
            prompt_length = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            outputs.extend(self._clean(text) for text in decoded)
        return outputs

    @staticmethod
    def _clean(text: str) -> str:
        """剥掉模型可能附带的"输出："前缀、引号与换行，只保留第一行。"""

        value = text.strip().split("\n")[0].strip()
        value = re.sub(r"^(?:输出|结果)\s*[:：]\s*", "", value)
        value = value.strip("“”\"'` \t")
        value = re.sub(r"\s+", "", value)
        return value[:MAX_OUTPUT_CHARS]


__all__ = ["QwenPurifier"]
