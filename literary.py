# literary.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Load once (important for performance)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",          # Spaces default
    torch_dtype=torch.float32  # safest on CPU
)

model.eval()


def literary_refine(
    source_lang: str,
    target_lang: str,
    source_text: str,
    draft_translation: str,
) -> str:
    """
    Uses LLaMA to rewrite a translation in a more fluent / literary way.
    """

    prompt = f"""
You are a professional literary translator.

Source language: {source_lang}
Target language: {target_lang}

Original text:
{source_text}

Draft translation:
{draft_translation}

Task:
Rewrite the draft translation to sound natural, fluent, and literary.
Preserve meaning. Do NOT add new content.
Return ONLY the improved translation.
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt from the output
    refined = decoded.replace(prompt, "").strip()

    # Safety fallback
    return refined if refined else draft_translation
