# -*- coding: utf-8 -*-
"""
Urdu Grammar Correction - Gemini AI + RAG + mT5 fallback
"""
import os
import gradio as gr
import torch

# ─── mT5 (fallback model) ───
MODEL_PATH = "./urdu_gec_model/final"
BASE_MODEL = "google/mt5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading mT5 fallback model...")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = model.to(device).eval()
print("mT5 ready")

# ─── Rule-based post-processor ───
from grammar_rules_post import apply_rules

# ─── RAG Retriever ───
retriever = None
try:
    from rag_retriever import UrduGrammarRetriever
    retriever = UrduGrammarRetriever()
    print("RAG retriever ready")
except Exception as e:
    print(f"No RAG retriever: {e}")

PREFIX = "correct grammar: "

# ─── Load .env ───
from dotenv import load_dotenv
load_dotenv()

# ─── Gemini API ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_CLIENT = None
try:
    from google import genai as gemini_sdk
except ImportError:
    gemini_sdk = None


def get_gemini_client():
    global GEMINI_CLIENT
    if not gemini_sdk or not GEMINI_API_KEY:
        return None
    if GEMINI_CLIENT is None:
        try:
            GEMINI_CLIENT = gemini_sdk.Client(api_key=GEMINI_API_KEY)
        except Exception:
            return None
    return GEMINI_CLIENT


def build_rag_context(text):
    """Build grammar rules context from RAG for the prompt."""
    if not retriever:
        return ""
    try:
        results = retriever.retrieve(text, top_k=4)
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            rule = r["rule"]
            parts.append(
                f"{i}. {rule['rule_en']}\n"
                f"   Wrong: {rule['example_wrong']}\n"
                f"   Right: {rule['example_right']}\n"
                f"   Note: {rule['explanation']}"
            )
        return "\n\n".join(parts)
    except Exception:
        return ""


def correct_with_gemini(text, mT5_output, rag_context):
    """Use Gemini to improve mT5's correction using RAG rules."""
    client = get_gemini_client()
    if client is None:
        return None

    rules_section = ""
    if rag_context:
        rules_section = f"\nRelevant Urdu Grammar Rules:\n{rag_context}\n"

    prompt = f"""You are an expert Urdu grammar proofreader. An AI model has attempted to correct this Urdu sentence. Your job is to review and improve the correction using the grammar rules provided.

Original (incorrect): {text}
Model's correction: {mT5_output}

{rules_section}
Instructions:
- Review the model's correction against the grammar rules above
- Fix any remaining errors the model missed (especially gender/number agreement, word order, spelling)
- Do NOT change the sentence if it's already correct
- Return ONLY the final corrected sentence, no explanations, no markdown

Final corrected sentence:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gemini_sdk.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=256,
            )
        )
        result = response.text.strip()
        for prefix in ["Final corrected sentence:", "Corrected:", "Output:", "Correct:"]:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        result = result.strip('\u201c\u201d"\'').strip("\u06d4").strip()
        return result if result and result != mT5_output else None
    except Exception:
        return None


def correct_with_mt5(text):
    """Fallback: use mT5 model."""
    input_text = PREFIX + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_length=256, num_beams=4,
            early_stopping=True, no_repeat_ngram_size=3
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ─── Main correction function ───
def correct(text, rag_mode):
    if not text or not text.strip():
        return "", "", ""

    text = text.strip()
    engine_used = ""

    # Determine what's enabled
    use_api_rag = rag_mode == "RAG Correction"
    use_rules_display = rag_mode != "No RAG"

    # 1. Build RAG context (for display + API prompt)
    rag_context = build_rag_context(text) if use_rules_display else ""

    # 2. ALWAYS run mT5 first
    mT5_output = correct_with_mt5(text)
    correction = mT5_output
    engine_used = "mT5-small + LoRA"

    # 3. Optionally refine with API-based RAG
    if use_api_rag:
        gemini_refined = correct_with_gemini(text, mT5_output, rag_context)
        if gemini_refined:
            correction = gemini_refined
            engine_used = "mT5 -> API-based RAG Refinement"
    elif rag_mode == "Only Display Rules":
        engine_used = "mT5 + RAG Rules (no API)"

    # 4. Apply rule-based post-processing (gender/number agreement)
    correction = apply_rules(correction)

    # 5. Format RAG rules for display (only if RAG enabled)
    rules_display = ""
    if use_rules_display and rag_context and retriever:
        results = retriever.retrieve(text, top_k=3)
        if results:
            rules_display = "### Grammar Rules Used\n\n"
            for i, r in enumerate(results, 1):
                rule = r["rule"]
                rules_display += f"**{i}. {rule['rule_en']}**\n\n"
                rules_display += f"Wrong: `{rule['example_wrong']}`\n\n"
                rules_display += f"Right: `{rule['example_right']}`\n\n"
                rules_display += f"*{rule['explanation']}*\n\n---\n\n"

    engine_info = f"Engine: **{engine_used}**"

    return correction, rules_display, engine_info


# ─── UI ───
EXAMPLES = [
    ("میں کل بازار گیا تھا سبزیاں خریدنے کے لیے", "Word Order"),
    ("وہ لڑکی بہت اچھا گاتی ہیں", "Verb Agreement"),
    ("اس نے مجے فون کیا", "Spelling"),
    ("بچے باہر کھیل رہے ہے", "Plural Verb"),
    ("وہ گھر گیا اور اور پھر واپس آیا", "Extra Words"),
    ("کل رات بجلی چلی گئ", "Spelling"),
    ("میں نے بازار سبزی خریدی", "Missing Words"),
    ("ہم کل بازار گئے تھا", "Gender Agreement"),
]

CSS = """
body, .gradio-container { background: #0b0f19 !important; color: #e2e8f0 !important; }
.block { background: transparent !important; }
.gr-group { background: #131a2e !important; border: 1px solid #1e293b !important; border-radius: 12px !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border: none !important; font-weight: 700 !important;
    border-radius: 10px !important; padding: 14px 24px !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.35) !important;
}
.gr-button-primary:hover { transform: translateY(-2px); }
.gr-button-secondary {
    background: transparent !important; border: 2px solid #334155 !important;
    color: #94a3b8 !important; font-weight: 600 !important; border-radius: 10px !important;
}
.gr-button-secondary:hover { border-color: #64748b !important; color: #e2e8f0 !important; }
.gr-slider { accent-color: #6366f1 !important; }
textarea {
    background: #1a2236 !important; border: 2px solid #1e293b !important;
    border-radius: 10px !important; color: #fbbf24 !important;
    font-size: 1.25rem !important; padding: 14px !important; line-height: 1.8 !important;
}
textarea:focus { border-color: #3b82f6 !important; }
label { color: #94a3b8 !important; font-weight: 600 !important; }
.gradio-container .prose { color: #e2e8f0 !important; }
"""

with gr.Blocks(title="Urdu Grammar Corrector") as app:
    gr.HTML("""
    <div style="text-align:center;padding:28px;background:linear-gradient(180deg,#1e3a5f,#0b0f19);border-radius:12px;margin-bottom:16px;border:1px solid #1e293b">
      <h1 style="color:white;margin:0;font-size:2rem">Urdu Grammar Corrector</h1>
      <p style="color:#94a3b8;margin:6px 0 0 0">AI-powered grammar correction for Urdu text with API-based RAG</p>
    </div>
    """)

    with gr.Group():
        with gr.Row():
            rag_mode = gr.Radio(
                choices=["RAG Correction", "Only Display Rules", "No RAG"],
                value="RAG Correction",
                label="RAG Mode",
            )
            engine_info = gr.Markdown("")

    with gr.Group():
        gr.Markdown("**Enter Urdu Text**")
        input_text = gr.Textbox(placeholder="Type Urdu sentence here...", lines=3, label="Incorrect Sentence")

        with gr.Row():
            btn = gr.Button("Correct Grammar", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

    with gr.Group():
        gr.Markdown("**Corrected Output**")
        output_text = gr.Textbox(lines=3, interactive=False, label="Corrected Sentence")

    with gr.Group():
        rag_output = gr.Markdown()

    gr.Markdown("**Examples**")
    with gr.Row():
        for ex_text, ex_label in EXAMPLES:
            ex_btn = gr.Button(ex_label, variant="secondary", size="sm")
            ex_btn.click(fn=lambda t=ex_text: t, inputs=[], outputs=[input_text])

    btn.click(
        fn=correct,
        inputs=[input_text, rag_mode],
        outputs=[output_text, rag_output, engine_info]
    )
    input_text.submit(
        fn=correct,
        inputs=[input_text, rag_mode],
        outputs=[output_text, rag_output, engine_info]
    )
    clear_btn.click(fn=lambda: ("", "", ""), inputs=[], outputs=[input_text, output_text, rag_output])

    gr.HTML('<div style="text-align:center;padding:20px;color:#475569;font-size:0.8rem">mT5-small + LoRA + API-based RAG + Rule-based post-processing</div>')

app.launch(server_port=7862, inbrowser=True, css=CSS)
