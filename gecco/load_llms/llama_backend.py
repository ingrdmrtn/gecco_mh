from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch

def load_llama(model_name: str):
    print(f"[GeCCo] Loading LLaMA tokenizer: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[GeCCo] Tokenizer loaded in {time.time() - t0:.1f}s")

    print(f"[GeCCo] Loading LLaMA model weights: {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
    )
    print(f"[GeCCo] Model loaded in {time.time() - t0:.1f}s (device_map=auto)")
    return tokenizer, model
