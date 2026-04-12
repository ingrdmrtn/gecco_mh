from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch

from gecco.utils import log as _log


def load_r1(model_name: str):
    _log(f"[GeCCo] Loading R1-Distilled tokenizer: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _log(f"[GeCCo] Tokenizer loaded in {time.time() - t0:.1f}s")

    _log(f"[GeCCo] Loading R1-Distilled model weights: {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
    )
    _log(f"[GeCCo] Model loaded in {time.time() - t0:.1f}s (device_map=auto)")
    return tokenizer, model
