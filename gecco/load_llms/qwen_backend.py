from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import time
import torch


def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def load_qwen(model_name: str):
    _log(f"[GeCCo] Loading Qwen tokenizer: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _log(f"[GeCCo] Tokenizer loaded in {time.time() - t0:.1f}s")

    _log(f"[GeCCo] Loading Qwen model weights: {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    _log(f"[GeCCo] Model loaded in {time.time() - t0:.1f}s (device_map=auto)")
    return tokenizer, model
