import gradio as gr
from transformers import pipeline
import torch
import datetime
import csv
import os
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
    user_msg = literary_user_prompt(source_lang_name, target_lang_name, source_text, draft_translation)

    messages = [
        {"role": "system", "content": LITERARY_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    # Qwen chat template formats messages properly
    prompt = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = llm_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = llm_model.generate(
            **inputs,
            max_new_tokens=256,     # bump to 512 for longer passages
            do_sample=False,        # deterministic
            temperature=0.0,
            repetition_penalty=1.05
        )

    text = llm_tokenizer.decode(out[0], skip_special_tokens=True)

    # Extract only the generated part after the input prompt
    answer = text[len(llm_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    answer = answer.strip()

    # Safety fallback: if extraction fails, return the draft
    return answer if answer else draft_translation


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
    else:
        final = literary_refine(src_lang, tgt_lang, text, draft)

    if show_both:
        return draft, final
    return final

# -----------------------------
# LOGGING FUNCTION
# -----------------------------
FEEDBACK_FILE = "ab_feedback.csv"

def log_ab_feedback(choice, src_lang, tgt_lang):
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "choice", "source_lang", "target_lang"])
        writer.writerow([
            datetime.datetime.utcnow().isoformat(),
            choice,
            src_lang,
            tgt_lang
        ])

    return "Feedback recorded. Thank you!"

# -----------------------------
# 4) Gradio UI
# -----------------------------
with gr.Blocks(title="Nepali ↔ English Translator (Side-by-Side)") as demo:
    gr.Markdown("# Nepali ↔ English Translator")
    gr.Markdown("Fast = NLLB translation, Literary = optional refined rewrite.")

    # Input row
    input_text = gr.Textbox(
        label="Input Text",
        placeholder="Enter text here...",
        lines=6
    )

    # Options
    with gr.Row():
        src_lang = gr.Dropdown(list(LANGS.keys()), value="English", label="Source")
        tgt_lang = gr.Dropdown(list(LANGS.keys()), value="Nepali", label="Target")

    use_literary = gr.Checkbox(
        label="Enable Literary Rewrite (slower)",
        value=False
    )

    # Output side-by-side
    with gr.Row():
        out_nllb = gr.Textbox(
            label="NLLB Translation (Accurate)",
            lines=8,
            interactive=False
        )
        out_literary = gr.Textbox(
            label="Literary Rewrite (if enabled)",
            lines=8,
            interactive=False
        )

    translate_btn = gr.Button("Translate")

    def side_by_side_translate(text, s, t, use_lit):
        if not text or not text.strip():
            return "", ""
        # 1) Always NLLB draft
        draft = nllb_translate(text, LANGS[s], LANGS[t])
        # 2) Only run LLM if enabled
        if use_lit:
            refined = literary_refine(s, t, text, draft)
        else:
            refined = ""
        return draft, refined

        gr.Markdown("### Which output do you prefer?")

    with gr.Row():
        btn_nllb = gr.Button("👍 Prefer NLLB")
        btn_llm = gr.Button("👍 Prefer Literary")
        btn_tie = gr.Button("🤝 Tie")
    
    feedback_status = gr.Textbox(
        label="Feedback status",
        interactive=False
    )

    translate_btn.click(
        side_by_side_translate,
        inputs=[input_text, src_lang, tgt_lang, use_literary],
        outputs=[out_nllb, out_literary]
    )

    btn_nllb.click(
        fn=lambda s=src_lang.value, t=tgt_lang.value: log_ab_feedback("NLLB", s, t),
        outputs=feedback_status
    )
    
    btn_llm.click(
        fn=lambda s=src_lang.value, t=tgt_lang.value: log_ab_feedback("LITERARY", s, t),
        outputs=feedback_status
    )
    
    btn_tie.click(
        fn=lambda s=src_lang.value, t=tgt_lang.value: log_ab_feedback("TIE", s, t),
        outputs=feedback_status
    )


demo.launch()
