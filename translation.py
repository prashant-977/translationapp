# translation.py
import torch
from transformers import pipeline

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

DEVICE = 0 if torch.cuda.is_available() else -1

translator = pipeline(
    "translation",
    model=NLLB_MODEL,
    device=DEVICE,
)

def nllb_translate(text: str, src_code: str, tgt_code: str) -> str:
    out = translator(
        text,
        src_lang=src_code,
        tgt_lang=tgt_code,
        max_length=512,
    )
    return out[0]["translation_text"]
