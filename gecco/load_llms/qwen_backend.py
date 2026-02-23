from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch

def load_qwen(model_name: str):
    print(f"[GeCCo] Loading Qwen tokenizer: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"[GeCCo] Tokenizer loaded in {time.time() - t0:.1f}s")

    print(f"[GeCCo] Loading Qwen model weights: {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[GeCCo] Model loaded in {time.time() - t0:.1f}s (device_map=auto)")
    return tokenizer, model
