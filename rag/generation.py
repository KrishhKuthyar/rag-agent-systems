"""Generate answers locally and check their source-label references."""

from pathlib import Path
import re
from typing import Any

from rag.prompting import INSUFFICIENT_EVIDENCE


ANSWER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ANSWER_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_INPUT_TOKENS = 2048


def load_answer_model(cache_dir: Path, allow_download: bool = False) -> tuple[Any, Any]:
    """Load a pinned tokenizer and CPU model, using cached files by default."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "revision": ANSWER_MODEL_REVISION,
        "cache_dir": str(cache_dir),
        "local_files_only": not allow_download,
        "token": False,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL, **settings)
    model = AutoModelForCausalLM.from_pretrained(
        ANSWER_MODEL, **settings, use_safetensors=True, dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_answer(
    messages: list[dict[str, str]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 128,
) -> str:
    """Generate one greedy answer without truncating the supplied prompt."""
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 256
    ):
        raise ValueError("max_new_tokens must be an integer between 1 and 256")
    if not messages:
        raise ValueError("answer messages must not be empty")

    import torch

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", truncation=False,
    )
    input_length = inputs["input_ids"].shape[-1]
    if input_length > MAX_INPUT_TOKENS:
        raise ValueError(f"prompt has {input_length} tokens; limit is {MAX_INPUT_TOKENS}")
    if input_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt and requested answer exceed the model context window")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_ids = outputs[0][input_length:].tolist()
    stop_ids = model.generation_config.eos_token_id
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    if len(answer_ids) >= max_new_tokens and answer_ids[-1] not in (stop_ids or []):
        raise ValueError("answer hit its token limit before completion; increase --max-new-tokens")
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    if not answer:
        raise ValueError("model returned an empty answer")
    return answer


def check_answer_citations(answer: str, source_count: int) -> None:
    """Check literal citation labels, without claiming to verify factual support."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a nonblank string")
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 0:
        raise ValueError("source_count must be a nonnegative integer")
    if answer.strip() == INSUFFICIENT_EVIDENCE:
        return
    labels = re.findall(r"\[S([^\]\n]*)\]", answer)
    if not labels:
        raise ValueError("generated answer has no source citations")
    allowed = {str(number) for number in range(1, source_count + 1)}
    for label in labels:
        if label not in allowed:
            raise ValueError(f"generated answer cites an unavailable source: [S{label}]")
