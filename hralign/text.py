from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .models import extract_model_state, load_torch_checkpoint


def load_r3m_language_state(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = load_torch_checkpoint(path)
    source = extract_model_state(checkpoint)
    result = {}
    for key, value in source.items():
        normalized = key.removeprefix("module.")
        if normalized.startswith("lang_enc.model."):
            result[normalized] = value
    if len(result) != 100:
        raise RuntimeError(
            f"Expected 100 R3M DistilBERT tensors in {path}; found {len(result)}."
        )
    return result


class FrozenR3MTextEncoder(nn.Module):
    """DistilBERT text encoder matching the original R3M implementation."""

    def __init__(
        self,
        r3m_language_state: Mapping[str, torch.Tensor],
        tokenizer_name: str = "distilbert-base-uncased",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        cache_by_text: bool = False,
    ):
        super().__init__()
        try:
            from transformers import (
                AutoTokenizer,
                DistilBertConfig,
                DistilBertModel,
            )
        except ImportError as error:
            raise ImportError(
                "HR-Align language conditioning requires transformers. "
                "Install the repository requirements first."
            ) from error

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = DistilBertModel(DistilBertConfig())
        model_state = {
            key.removeprefix("lang_enc.model."): value
            for key, value in r3m_language_state.items()
        }
        missing, unexpected = self.model.load_state_dict(
            model_state, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                "R3M DistilBERT state is incompatible: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.cache_by_text = cache_by_text
        self._cache: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def _encode_batch(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = self.model(**encoded).last_hidden_state
        # This intentionally matches R3M's released LangEncoder, which averages
        # the padded sequence dimension without an attention-mask reduction.
        return hidden.mean(dim=1)

    @torch.no_grad()
    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        if not self.cache_by_text:
            return self._encode_batch(texts)

        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        for text in missing:
            self._cache[text] = self._encode_batch([text])[0].cpu()
        device = next(self.model.parameters()).device
        return torch.stack([self._cache[text] for text in texts]).to(device)


class PrecomputedTaskTextEncoder(nn.Module):
    """Loads deterministic task embeddings for offline or fast training."""

    def __init__(self, path: str | Path):
        super().__init__()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        embeddings = payload.get("embeddings", payload)
        if not isinstance(embeddings, dict):
            raise TypeError("Precomputed text embeddings must be a dictionary.")
        self.embeddings = {
            str(key): torch.as_tensor(value, dtype=torch.float32)
            for key, value in embeddings.items()
        }
        if not self.embeddings:
            raise ValueError("Precomputed text embedding file is empty.")
        dimensions = {value.shape for value in self.embeddings.values()}
        if dimensions != {torch.Size([768])}:
            raise ValueError(
                f"Expected 768-D task embeddings; found shapes {dimensions}."
            )
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def forward(
        self, task_ids: Sequence[str], texts: Sequence[str] | None = None
    ) -> torch.Tensor:
        del texts
        missing = [task_id for task_id in task_ids if task_id not in self.embeddings]
        if missing:
            raise KeyError(f"Missing precomputed task embeddings: {missing[:10]}")
        return torch.stack(
            [self.embeddings[task_id] for task_id in task_ids]
        ).to(self._device_anchor.device)
