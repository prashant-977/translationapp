# feedback.py
import csv
import os
import datetime

FEEDBACK_FILE = "ab_feedback.csv"

def log_feedback(model_name, sentiment, src_lang, tgt_lang):
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "model",
                "sentiment",
                "source_lang",
                "target_lang",
            ])

        writer.writerow([
            datetime.datetime.utcnow().isoformat(),
            model_name,
            sentiment,
            src_lang,
            tgt_lang,
        ])

    return "Feedback recorded. Thank you!"
