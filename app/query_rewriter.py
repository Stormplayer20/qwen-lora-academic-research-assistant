"""
Fine-tuned academic query rewriter for the AI Research Assistant.

Uses:
    Qwen/Qwen2.5-0.5B-Instruct
    +
    a custom LoRA adapter trained on academic search-query rewriting examples.

The base Qwen model provides the language-model capabilities, while the
LoRA adapter contains the task-specific fine-tuning learned from our
research-query dataset.

The resulting model rewrites informal research topics into concise,
keyword-rich queries that are sent to the Semantic Scholar API.
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
from peft import PeftModel # type: ignore

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "research-assistant-model"

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
        print("Loading base Qwen model...")

        _tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
        )

        print("Applying fine-tuned LoRA adapter...")

        _model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH),
        )

        _model.to("cpu")
        _model.eval()

        _model.generation_config.do_sample = False
        _model.generation_config.temperature = None
        _model.generation_config.top_p = None
        _model.generation_config.top_k = None

        print("Fine-tuned model loaded successfully.")

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
