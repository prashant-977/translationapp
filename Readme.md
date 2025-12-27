# Nepali-English Literary Translation App - Development Plan

**Project Goal:** Build a translation app that preserves literary richness, puns, and cultural nuance for Nepali-English translations using free Hugging Face resources.

---

# 🎯 Project Overview

> **Challenge:** Standard translation models flatten jokes and metaphors into literal meanings. This project aims to preserve the "voice" and "style" of literary texts through instruction-tuned LLMs and specialized translation models.
> 

**Key Differentiator:** Moving beyond functional translation to capture cultural context, humor, and literary style.

---

# 🧠 Model Selection Strategy

### Phase 1: MVP (Base Translation)

**Primary Model:** `facebook/nllb-200-distilled-600M`

**Rationale:**

- Lightweight and fits in free tier
- State-of-the-art for low-resource languages like Nepali
- Excellent grammatical accuracy
- Fast inference time

**Trade-off:** May miss puns and literary nuance without fine-tuning

### Phase 2: Literary Enhancement

**Primary Model:** `meta-llama/Llama-3.2-3B-Instruct` or `MISHANM/Nepali_NLP_eng_to_nepali_Llama3.2_3B_instruction`

**Rationale:**

- Generative LLM allows custom prompting
- Can preserve "voice" and "style" through instructions
- Capable of creative adaptation of puns and idioms

**Implementation Approach:**

```python
system_prompt = "You are a professional literary translator. Translate the following text into Nepali. If you encounter a pun or idiom, adapt it creatively so it is funny or meaningful for a Nepali reader, rather than translating literally."
```

### Alternative Option

**Model:** `jbochi/madlad400-3b-mt`

- Google research model trained on 3 trillion tokens
- Strong multilingual translation
- Often outperforms NLLB in fluency

---

# 📊 Dataset Strategy

### Base Dataset

**Source:** `ashokpoudel/nepali-english-translation-dataset`

- Scale: 1M-10M sentence pairs
- **Strength:** Teaches grammar and basic structure
- **Limitation:** Primarily news and technical content, not literary

### Literary Enhancement Dataset

**Strategy:** Create or curate a specialized literary dataset

**Approach:**

- Manually align 500-1000 high-quality English-Nepali sentence pairs from novels and poems
- Focus on:
    - Metaphors and figurative language
    - Humor and wordplay
    - Cultural idioms
    - Stylistic variations

**Impact:** Even 500 high-quality literary pairs (fine-tuned with LoRA) will be more effective than 1M news sentences for literary style.

### Data Sources to Consider

- [ ]  Nepali literary classics in public domain
- [ ]  Bilingual poetry collections
- [ ]  Translated novels with both versions available
- [ ]  Folk tales and proverbs
- [ ]  Contemporary Nepali literature with English translations

---

# 🏗️ Development Roadmap

## Phase 1: MVP Development (Weeks 1-2)

### Week 1: Setup & Basic Implementation

- [ ]  Set up Hugging Face Space account
- [ ]  Install dependencies (transformers, gradio, torch)
- [ ]  Implement basic NLLB-200 translation pipeline
- [ ]  Create simple Gradio interface
- [ ]  Test with sample sentences

### Week 2: MVP Refinement

- [ ]  Add language detection
- [ ]  Implement bidirectional translation (EN→NP and NP→EN)
- [ ]  Add input validation and error handling
- [ ]  Create user-friendly interface with examples
- [ ]  Deploy initial version to Hugging Face Space

## Phase 2: Literary Enhancement (Weeks 3-5)

### Week 3: LLM Integration

- [ ]  Request access to Llama 3.2 on Hugging Face
- [ ]  Implement LLM-based translation with literary prompt
- [ ]  Create A/B testing interface (NLLB vs Llama)
- [ ]  Test with literary samples (poems, novel excerpts)

### Week 4: Dataset Curation

- [ ]  Identify literary sources (novels, poems, folk tales)
- [ ]  Manually curate 100 high-quality translation pairs
- [ ]  Annotate for puns, idioms, and cultural references
- [ ]  Create evaluation metrics for literary quality

### Week 5: Fine-tuning Preparation

- [ ]  Complete dataset to 500+ pairs
- [ ]  Split into train/validation/test sets
- [ ]  Set up LoRA fine-tuning pipeline
- [ ]  Document data collection process

## Phase 3: Fine-tuning & Optimization (Weeks 6-8)

### Week 6: Model Fine-tuning

- [ ]  Fine-tune NLLB with literary dataset using LoRA
- [ ]  Monitor training metrics
- [ ]  Evaluate on validation set
- [ ]  Compare with base model performance

### Week 7: Quality Evaluation

- [ ]  Create evaluation framework:
    - Grammatical accuracy
    - Literary style preservation
    - Pun/idiom adaptation quality
    - Cultural appropriateness
- [ ]  Test with native Nepali speakers
- [ ]  Collect feedback and iterate

### Week 8: Optimization

- [ ]  Optimize model for inference speed
- [ ]  Implement caching for common phrases
- [ ]  Add batch processing capability
- [ ]  Performance testing and profiling

## Phase 4: Advanced Features (Weeks 9-10)

### Week 9: Feature Enhancement

- [ ]  Add context preservation (translate full paragraphs)
- [ ]  Implement style controls (formal/informal, poetic/prose)
- [ ]  Add translation confidence scores
- [ ]  Create "explanation mode" (why certain adaptations were made)

### Week 10: Polish & Launch

- [ ]  Final UI/UX improvements
- [ ]  Add example translations showcasing literary preservation
- [ ]  Create documentation and user guide
- [ ]  Launch public demo
- [ ]  Gather user feedback

---

# 💻 Technical Implementation

## MVP Code Structure

### Basic NLLB Implementation

```python
import gradio as gr
from transformers import pipeline

# Load NLLB model (optimized for translation)
translator = pipeline(
    "translation", 
    model="facebook/nllb-200-distilled-600M", 
    src_lang="eng_Latn", 
    tgt_lang="npi_Deva"
)

def translate_literary(text):
    result = translator(text)
    return result[0]['translation_text']

# Create the User Interface
interface = gr.Interface(
    fn=translate_literary,
    inputs=gr.Textbox(lines=5, placeholder="Enter text from a book..."),
    outputs="text",
    title="Nepali-English Literary Translator",
    description="An AI translator focusing on literature."
)

interface.launch()
```

### Enhanced LLM Implementation

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Llama 3.2 model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def translate_with_literary_context(text, preserve_style=True):
    system_prompt = (
        "You are a professional literary translator specializing in "
        "English-Nepali translation. Translate the following text into Nepali. "
        "If you encounter a pun or idiom, adapt it creatively so it is funny "
        "or meaningful for a Nepali reader, rather than translating literally. "
        "Preserve the tone, style, and emotional impact of the original."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate to Nepali: {text}"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt"
    )
    
    outputs = model.generate(inputs, max_new_tokens=512)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation

interface = gr.Interface(
    fn=translate_with_literary_context,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter literary text..."),
        gr.Checkbox(label="Preserve literary style", value=True)
    ],
    outputs="text",
    title="Literary Translator (Advanced)",
    description="Preserves puns, metaphors, and cultural nuance"
)

interface.launch()
```

---

# 📈 Success Metrics

### Quantitative Metrics

- **BLEU Score:** Baseline translation accuracy
- **chrF Score:** Character-level accuracy (better for morphologically rich languages)
- **BERTScore:** Semantic similarity
- **Inference Time:** < 2 seconds per sentence on free tier

### Qualitative Metrics

- **Literary Style Preservation:** Human evaluation (1-5 scale)
- **Pun/Idiom Adaptation Quality:** Native speaker ratings
- **Cultural Appropriateness:** Feedback from Nepali readers
- **User Satisfaction:** NPS score from demo users

---

# 🔗 Key Resources

### Models

- **NLLB-200 Distilled 600M:** [https://huggingface.co/facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **Llama-3.2-3B-Instruct:** [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **MADLAD400-3B-MT:** `jbochi/madlad400-3b-mt`
- **Nepali-specific Llama:** `MISHANM/Nepali_NLP_eng_to_nepali_Llama3.2_3B_instruction`

### Datasets

- **Base Dataset:** [https://www.google.com/search?q=https://huggingface.co/datasets/ashokpoudel/nepali-english-translation-dataset](https://www.google.com/search?q=https://huggingface.co/datasets/ashokpoudel/nepali-english-translation-dataset)
- **Literary Dataset:** To be curated

### Documentation

- NLLB Architecture: [https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DpM9U59rXlJE](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DpM9U59rXlJE)
- Hugging Face Spaces: [https://huggingface.co/spaces](https://huggingface.co/spaces)
- Gradio Documentation: [https://gradio.app/docs](https://gradio.app/docs)
- LoRA Fine-tuning: PEFT library documentation

---

# 🚀 Deployment Strategy

### Free Tier Deployment (Hugging Face Spaces)

**Advantages:**

- Completely free hosting
- Gradio integration
- Easy sharing and collaboration
- GPU access (limited but sufficient for inference)

**Limitations:**

- 16GB RAM limit
- 50GB storage
- Shared GPU (slower inference during peak times)

**Optimization Strategies:**

- Use quantized models (8-bit or 4-bit) to reduce memory
- Implement response caching
- Use smaller models for MVP (600M parameters vs 3.3B)

### Alternative: Streamlit Cloud

- Similar free tier
- Different UI framework
- Good for rapid prototyping

---

# 🎓 Learning Resources

### Required Skills

- [ ]  Python programming
- [ ]  Transformers library (Hugging Face)
- [ ]  Gradio or Streamlit for UI
- [ ]  Basic understanding of NLP and translation models
- [ ]  Fine-tuning with LoRA (PEFT library)

### Recommended Tutorials

- Hugging Face NLP Course (free)
- NLLB architecture paper and implementation
- LoRA fine-tuning tutorials
- Nepali NLP resources and communities

---

# 🤝 Collaboration Opportunities

### Potential Partners

- Nepali language researchers
- Literary translators (human experts)
- Nepali literature communities
- Language preservation organizations

### Feedback Collection

- Create survey for native speakers
- A/B testing with different models
- Comparative evaluation (AI vs human translation)

---

# ⚠️ Challenges & Mitigation

### Challenge 1: Limited Literary Training Data

**Mitigation:** Manual curation of high-quality pairs + active learning from user feedback

### Challenge 2: Pun/Idiom Detection

**Mitigation:** Use LLM with explicit instructions to identify and adapt figurative language

### Challenge 3: Free Tier Resource Constraints

**Mitigation:** Start with distilled models, optimize inference, implement caching

### Challenge 4: Evaluation of Literary Quality

**Mitigation:** Combine automated metrics with human evaluation from native speakers

### Challenge 5: Cultural Context Differences

**Mitigation:** Build cultural knowledge base, consult with Nepali literature experts

---

# 📝 Next Steps

## Immediate Actions (This Week)

- [ ]  Create Hugging Face account
- [ ]  Set up development environment
- [ ]  Test NLLB-200-distilled-600M with sample texts
- [ ]  Create basic Gradio interface
- [ ]  Identify 3-5 literary sources for test cases

## Short-term Goals (This Month)

- [ ]  Deploy MVP to Hugging Face Spaces
- [ ]  Request access to Llama 3.2
- [ ]  Begin curating literary dataset (target: 100 pairs)
- [ ]  Test with 5-10 native Nepali speakers

## Long-term Vision

- Build the most culturally-aware Nepali-English literary translator
- Create open-source dataset for literary translation research
- Expand to other low-resource language pairs
- Collaborate with publishers for real-world applications
