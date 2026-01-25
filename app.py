# app.py
import gradio as gr

from translation import nllb_translate
from literary import literary_refine
from feedback import log_feedback


# -----------------------------
# Language configuration
# -----------------------------
LANGS = {
    "English": "eng_Latn",
    "Nepali": "npi_Deva",
}


# -----------------------------
# Translation logic
# -----------------------------
def side_by_side_translate(text, src_lang, tgt_lang, use_literary):
    if not text or not text.strip():
        return "", ""

    draft = nllb_translate(
        text,
        LANGS[src_lang],
        LANGS[tgt_lang],
    )

    refined = (
        literary_refine(src_lang, tgt_lang, text, draft)
        if use_literary
        else ""
    )

    return draft, refined


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Nepali ↔ English Translator (Side-by-Side)") as demo:
    gr.Markdown("# Nepali ↔ English Translator")
    gr.Markdown("**NLLB = accurate | Literary = optional refined rewrite**")

    # Input
    input_text = gr.Textbox(
        label="Input Text",
        placeholder="Enter text here...",
        lines=6,
    )

    with gr.Row():
        src_lang = gr.Dropdown(
            choices=list(LANGS.keys()),
            value="English",
            label="Source Language",
        )
        tgt_lang = gr.Dropdown(
            choices=list(LANGS.keys()),
            value="Nepali",
            label="Target Language",
        )

    use_literary = gr.Checkbox(
        label="Enable Literary Rewrite (slower)",
        value=False,
    )

    # Outputs
    with gr.Row():
        out_nllb = gr.Textbox(
            label="NLLB Translation (Accurate)",
            lines=8,
            interactive=False,
        )
        out_literary = gr.Textbox(
            label="Literary Rewrite",
            lines=8,
            interactive=False,
        )

    translate_btn = gr.Button("Translate")

    translate_btn.click(
        side_by_side_translate,
        inputs=[input_text, src_lang, tgt_lang, use_literary],
        outputs=[out_nllb, out_literary],
    )

    # -----------------------------
    # Feedback section
    # -----------------------------
    gr.Markdown("### Rate each output")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**NLLB Translation**")
            btn_nllb_up = gr.Button("👍 Good")
            btn_nllb_down = gr.Button("👎 Bad")

        with gr.Column():
            gr.Markdown("**Literary Rewrite**")
            btn_llm_up = gr.Button("👍 Good")
            btn_llm_down = gr.Button("👎 Bad")

    feedback_status = gr.Textbox(
        label="Feedback status",
        interactive=False,
    )

    # Feedback wiring
    btn_nllb_up.click(
        log_feedback,
        inputs=[gr.State("NLLB"), gr.State("UP"), src_lang, tgt_lang],
        outputs=feedback_status,
    )

    btn_nllb_down.click(
        log_feedback,
        inputs=[gr.State("NLLB"), gr.State("DOWN"), src_lang, tgt_lang],
        outputs=feedback_status,
    )

    btn_llm_up.click(
        log_feedback,
        inputs=[gr.State("LITERARY"), gr.State("UP"), src_lang, tgt_lang],
        outputs=feedback_status,
    )

    btn_llm_down.click(
        log_feedback,
        inputs=[gr.State("LITERARY"), gr.State("DOWN"), src_lang, tgt_lang],
        outputs=feedback_status,
    )

demo.launch()
