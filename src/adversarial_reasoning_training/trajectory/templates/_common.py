"""Chat template markers shared across the Qwen and InternVL2 assemblers.

Qwen2.5-VL and InternVL2 both ride InternLM-style ``<|im_start|>`` /
``<|im_end|>`` role brackets — the latter inherits the convention from
InternLM2-Chat. Keeping the literals here avoids the cross-module
duplication the pre-split ``templates.py`` carried (it defined
``QWEN_IM_START`` once and InternVL2 re-used the symbol by name).

LLaVA-NeXT uses Mistral-Instruct ``[INST] ... [/INST]`` and has its own
markers in :mod:`._llava_next`.
"""

from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
