LITERARY_SYSTEM = """You are a professional literary translator.
Your job is to improve a draft translation while keeping meaning faithful.
Rules:
- Preserve meaning and facts; do not add new information.
- Keep tone, voice, and implied humor/sarcasm.
- Translate idioms naturally, not literally.
- Output ONLY the final translation text. No explanations."""

def literary_user_prompt(source_lang_name: str, target_lang_name: str, source_text: str, draft_translation: str) -> str:
    return f"""Source language: {source_lang_name}
Target language: {target_lang_name}

SOURCE TEXT:
{source_text}

DRAFT TRANSLATION:
{draft_translation}

Task: Rewrite the draft into a high-quality literary translation in {target_lang_name}.
Return only the improved translation."""