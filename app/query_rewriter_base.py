"""
Rewrites a raw, informal research question into a precise academic search query.

BASE VERSION (before fine-tuning): uses the base Qwen2.5-0.5B-Instruct model
with a system prompt. No fine-tuning needed yet -- this lets your pipeline
work end-to-end immediately.

MODEL SIZE NOTE: this defaults to the 0.5B model because it's fine-tunable on
CPU in a reasonable amount of time (see training/finetune_lora_cpu.py). If you
later get access to a GPU (e.g. free Google Colab), you can swap MODEL_PATH to
"Qwen/Qwen2.5-3B-Instruct" for noticeably better quality -- nothing else in
this file needs to change.

AFTER FINE-TUNING: change MODEL_PATH below to your saved adapter directory
(e.g. "research-assistant-model") produced by
training/finetune_lora_cpu.py. Everything else (main.py, ranker.py) stays
the same since the function signature doesn't change.
"""
import os
from pathlib import Path

# Store Hugging Face files inside this project's writable cache directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT / ".hf-cache"
HF_HOME.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_TOKEN_PATH", str(HF_HOME / "token"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # swap to "research-assistant-model" after fine-tuning

SYSTEM_PROMPT = (
    "You are an academic search-query optimizer. Convert the user's informal "
    "research topic into a concise, keyword-rich search query for an academic "
    "paper database. Preserve every important concept from the user's original "
    "query and add useful scholarly synonyms or closely related terminology. "
    "Return ONLY the search terms. Do not write a title, question, explanation, "
    "sentence, or preamble. Do not use quotation marks. Do not include phrases "
    "such as 'scholarly search query', 'literature review', or 'research on'. "
    "Prefer approximately 6-15 useful search terms."
)

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # float32 on CPU -- bfloat16/float16 don't reliably speed things up on
        # CPU and can be unsupported on some hardware, so plain float32 is the
        # safest default here.
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float32
        )
        _model.to("cpu")
        _model.eval()

        _model.generation_config.do_sample = False
        _model.generation_config.temperature = None
        _model.generation_config.top_p = None
        _model.generation_config.top_k = None
    return _tokenizer, _model


def rewrite_query(raw_query: str) -> str:
    tokenizer, model = _load()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_query},
    ]
    prompt = tokenizer.apply_chat_template( # type: ignore
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # type: ignore
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]

    rewritten = tokenizer.decode( # type: ignore
        generated,
        skip_special_tokens=True,
    ).strip()

    return rewritten.strip('"').strip("'").strip()

if __name__ == "__main__":
    # quick manual test: python -m app.query_rewriter
    print(rewrite_query("sleep academic performance college"))
