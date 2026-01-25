# literary.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1) Model setup
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Hugging Face token:
# - On HF Spaces: set as secret HF_TOKEN
# - Locally: can export as environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN not found. "
        "Set environment variable HF_TOKEN locally or add secret HF_TOKEN on Hugging Face Spaces."
    )

# -----------------------------
# 2) Load tokenizer and model
# -----------------------------
# Automatically selects GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.eval()

# -----------------------------
# 3) Literary refine function
# -----------------------------
def literary_refine(source_lang_name: str, target_lang_name: str, source_text: str, draft_translation: str) -> str:
    """
    Refines a draft translation using LLaMA 3.2 for more fluent / literary output.
    """
    # Construct a simple prompt
    prompt = (
        f"Refine the following English → Nepali translation to make it fluent and literary. "
        f"Ensure the output is in Nepali (Devanagari script), not Hindi or any other language.\n\n"
        f"Original Text:\n{source_text}\n\n"
        f"Draft Translation:\n{draft_translation}\n\n"
        "Refined Translation in Nepali:"
    )



    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Increase for longer text
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.05
        )

    refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from output
    refined_text = refined_text[len(prompt):].strip()

    return refined_text if refined_text else draft_translation
