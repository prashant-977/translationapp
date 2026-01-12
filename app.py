TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_TOKENS = 0
API_CALLS = 0

import gradio as gr
from transformers import pipeline
import torch
from prompts import LITERARY_SYSTEM, literary_user_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
client = OpenAI()

LITERARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# CPU-friendly loading
llm_tokenizer = AutoTokenizer.from_pretrained(LITERARY_MODEL, use_fast=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LITERARY_MODEL,
    device_map="cpu",          # force CPU for a laptop
    torch_dtype=torch.float32  # safest on CPU
)
llm_model.eval()

def literary_refine(source_lang_name: str, target_lang_name: str, source_text: str, draft_translation: str) -> str:
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_TOKENS, API_CALLS

    user_msg = literary_user_prompt(source_lang_name, target_lang_name, source_text, draft_translation)

    resp = client.responses.create(
        model="gpt-5.1-mini",
        input=[
            {"role": "system", "content": LITERARY_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )

    refined = (resp.output_text or "").strip()

    # ---- usage logging ----
    usage = getattr(resp, "usage", None)
    if usage:
        API_CALLS += 1
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        tot_tok = getattr(usage, "total_tokens", 0) or (in_tok + out_tok)

        TOTAL_INPUT_TOKENS += in_tok
        TOTAL_OUTPUT_TOKENS += out_tok
        TOTAL_TOKENS += tot_tok

        print(
            f"[OpenAI usage] input={in_tok} output={out_tok} total={tot_tok} | "
            f"session totals: calls={API_CALLS}, input={TOTAL_INPUT_TOKENS}, output={TOTAL_OUTPUT_TOKENS}, total={TOTAL_TOKENS}"
        )

    return refined if refined else draft_translation


# -----------------------------
# 1) NLLB setup
# -----------------------------
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

LANGS = {
    "English": "eng_Latn",
    "Nepali": "npi_Deva",
}

# Load NLLB translation pipeline once.
# device=-1 means CPU, device=0 means GPU if available.
DEVICE = 0 if torch.cuda.is_available() else -1

nllb_translator = pipeline(
    "translation",
    model=NLLB_MODEL,
    device=DEVICE,
)

def nllb_translate(text: str, src_code: str, tgt_code: str) -> str:
    # NLLB pipeline expects src_lang/tgt_lang in call kwargs for many setups
    out = nllb_translator(text, src_lang=src_code, tgt_lang=tgt_code, max_length=512)
    return out[0]["translation_text"]


# -----------------------------
# 2) Literary LLM setup
# -----------------------------

USE_LLM = True  # flip to True after you implement literary_refine properly

def literary_refine(source_lang_name: str, target_lang_name: str, source_text: str, draft_translation: str) -> str:
    user_msg = literary_user_prompt(source_lang_name, target_lang_name, source_text, draft_translation)

    # Responses API call (recommended for new projects)
    resp = client.responses.create(
        model="gpt-5.1-mini",  # good quality/price; you can change later
        input=[
            {"role": "system", "content": LITERARY_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )

    # The SDK provides output_text for the “just give me text” use case
    refined = (resp.output_text or "").strip()
    return refined if refined else draft_translation

# -----------------------------
# 3) App function (Fast vs Literary)
# -----------------------------
def translate_app(text: str, src_lang: str, tgt_lang: str, mode: str, show_both: bool):
    if not text or not text.strip():
        return "" if not show_both else ("", "")

    src_code = LANGS[src_lang]
    tgt_code = LANGS[tgt_lang]

    draft = nllb_translate(text, src_code, tgt_code)

    if mode == "Fast":
        final = draft
    else:  # Literary only
        final = literary_refine(src_lang, tgt_lang, text, draft)

    if show_both:
        return draft, final
    return final

# -----------------------------
# 4) Gradio UI
# -----------------------------
with gr.Blocks(title="Nepali ↔ English Translator") as demo:
    gr.Markdown("# Nepali ↔ English Translation (Fast vs Literary)")
    gr.Markdown("Fast = direct model translation. Literary = draft + stylistic refinement.")

    with gr.Row():
        src_lang = gr.Dropdown(choices=list(LANGS.keys()), value="English", label="Source")
        tgt_lang = gr.Dropdown(choices=list(LANGS.keys()), value="Nepali", label="Target")

    mode = gr.Radio(choices=["Fast", "Literary"], value="Fast", label="Mode")
    show_both = gr.Checkbox(value=False, label="Show both draft and final")

    inp = gr.Textbox(lines=8, label="Input Text", placeholder="Paste text here...")

    out_single = gr.Textbox(lines=8, label="Translation")
    out_draft = gr.Textbox(lines=8, label="Draft (NLLB)")
    out_final = gr.Textbox(lines=8, label="Final (Literary)")

    def route_outputs(text, s, t, m, both):
        result = translate_app(text, s, t, m, both)
        if both:
            draft, final = result
            return gr.update(visible=False, value=""), gr.update(visible=True, value=draft), gr.update(visible=True, value=final)
        else:
            return gr.update(visible=True, value=result), gr.update(visible=False, value=""), gr.update(visible=False, value="")

    btn = gr.Button("Translate")
    btn.click(route_outputs, inputs=[inp, src_lang, tgt_lang, mode, show_both], outputs=[out_single, out_draft, out_final])

demo.launch()
