# Nepali ↔ English Translator (NLLB + LLaMA)

A side-by-side Nepali ↔ English translation app that compares:

- **NLLB (facebook/nllb-200)** for accurate machine translation
- **LLaMA 3.2** for optional literary / fluent rewriting

The app is built with **Gradio**, runs **locally on CPU**, and can be deployed
unchanged to **Hugging Face Spaces**.

---

## Features

- Nepali ↔ English translation
- Fast, accurate baseline using NLLB
- Optional literary rewrite using LLaMA 3.2
- Side-by-side A/B comparison UI
- Per-model feedback (positive / negative)
- CSV logging for evaluation
- Works locally and on Hugging Face Spaces

---

## Architecture

User Input
↓
NLLB Translation (accurate)
↓
(optional)
LLaMA 3.2 Literary Rewrite


- **NLLB** handles meaning and correctness
- **LLaMA** improves fluency and style
- Users can compare and rate outputs

---

## Project Structure

.
├── app.py # Gradio UI
|
├── translation.py # NLLB translation logic
|
├── literary.py # LLaMA literary rewrite
|
├── feedback.py # Feedback logging (CSV)
|
├── requirements.txt
|
└── README.md

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

2️⃣ Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

### Local Setup

    Set your Hugging Face token (required for LLaMA 3.2):

    export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"


    Run the app:

    python app.py


Open the Gradio interface in your browser.

4️⃣ Hugging Face login (required for LLaMA)

LLaMA models are gated. You must:

    Request access to meta-llama/Llama-3.2-1B-Instruct

    Create a Hugging Face access token

    Enable “Read access to public gated repositories”

Then log in:

huggingface-cli login

5️⃣ Run the app

python app.py

Open the URL shown in the terminal (usually http://127.0.0.1:7860).
Deploy to Hugging Face Spaces

This repository can be deployed as-is.
Steps:

    Create a new Space

    Choose Gradio

    Push this repository to the Space

    Add a secret:

        Name: HF_TOKEN

        Value: your Hugging Face token

    Restart the Space

No code changes required.
Feedback Logging

User feedback is saved to:

feedback.csv

Each row includes:

    timestamp

    model (NLLB / LLaMA)

    rating (UP / DOWN)

    source language

    target language

On Hugging Face Spaces, the file is stored in the Space container.
Performance Notes

    First run downloads models (can take a few minutes)

    CPU inference is slower for literary rewrite

    NLLB translation remains fast

    LLaMA rewrite runs only when enabled

Notes

Literary refinement always outputs Nepali (Devanagari script)

For local runs, each user must provide their own HF_TOKEN with access to gated models

Feedback is logged locally to ab_feedback.csv