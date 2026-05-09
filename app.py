# -*- coding: utf-8 -*-
"""
Urdu Grammar Correction - Dark Modern Web UI
"""
import gradio as gr
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_PATH = "./urdu_gec_model/final"
BASE_MODEL = "google/mt5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = model.to(device).eval()
print("Model ready")

retriever = None
try:
    from rag_retriever import UrduGrammarRetriever
    retriever = UrduGrammarRetriever()
    print("Retriever ready")
except Exception as e:
    print(f"No retriever: {e}")

PREFIX = "correct grammar: "


def correct(text, beams, use_rag):
    try:
        text = text.strip()
        if not text:
            return "", ""

        input_text = PREFIX + text

        # Always use clean input - never inject RAG into prompt (confuses model)
        inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_length=256, num_beams=int(beams),
                early_stopping=True, no_repeat_ngram_size=3
            )
        correction = tokenizer.decode(out[0], skip_special_tokens=True)

        # RAG: retrieve rules for display only, not injected into model
        rules_md = ""
        if use_rag and retriever:
            _, retrieved = retriever.build_context(text, top_k=3)
            if retrieved:
                rules_md = "### Grammar Rules (Reference Only)\n\n"
                for i, r in enumerate(retrieved, 1):
                    rule = r["rule"]
                    rules_md += f"**{i}. {rule['rule_en']}**\n\n"
                    rules_md += f"Category: `{rule['category']}` | Error: `{rule['error_type']}`\n\n"
                    rules_md += f"❌ {rule['example_wrong']}  \n✅ {rule['example_right']}\n\n"
                    rules_md += f"*{rule['explanation']}*\n\n---\n\n"

        return correction, rules_md

    except Exception as e:
        return f"Error: {e}", ""


EXAMPLES = [
    ("میں کل بازار گیا تھا سبزیاں خریدنے کے لیے", "Word Order"),
    ("وہ لڑکی بہت اچھا گاتی ہیں", "Verb Agreement"),
    ("اس نے مجے فون کیا", "Spelling"),
    ("بچے باہر کھیل رہے ہے", "Plural Verb"),
    ("وہ گھر گیا اور اور پھر واپس آیا", "Extra Words"),
    ("کل رات بجلی چلی گئ", "Spelling"),
    ("میں نے بازار سبزی خریدی", "Missing Words"),
    ("وہ ہر روز صبح جانچ کرتہ ہے", "Spelling"),
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
textarea {
    background: #1a2236 !important; border: 2px solid #1e293b !important;
    border-radius: 10px !important; color: #fbbf24 !important;
    font-size: 1.25rem !important; padding: 14px !important; line-height: 1.8 !important;
}
textarea:focus { border-color: #3b82f6 !important; }
label { color: #94a3b8 !important; font-weight: 600 !important; }
"""


with gr.Blocks(css=CSS, title="Urdu Grammar Corrector") as app:
    gr.HTML("""
    <div style="text-align:center;padding:28px;background:linear-gradient(180deg,#1e3a5f,#0b0f19);border-radius:12px;margin-bottom:16px;border:1px solid #1e293b">
      <h1 style="color:white;margin:0;font-size:2rem">📝 Urdu Grammar Corrector</h1>
      <p style="color:#94a3b8;margin:6px 0 0 0">AI-powered grammar correction for Urdu text</p>
    </div>
    """)

    with gr.Group():
        gr.Markdown("**Enter Urdu Text**")
        input_text = gr.Textbox(placeholder="Type here...", lines=3, label="Incorrect Sentence")

        with gr.Row():
            btn = gr.Button("Correct Grammar", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Row():
            beams = gr.Slider(1, 8, value=4, step=1, label="Beam Width")
            use_rag = gr.Checkbox(True, label="Enable RAG")

    with gr.Group():
        gr.Markdown("**Corrected Output**")
        output_text = gr.Textbox(lines=3, interactive=False, label="Corrected Sentence")

    with gr.Group():
        gr.Markdown("**Grammar Rules**")
        rag_output = gr.Markdown()

    gr.Markdown("**Examples**")
    with gr.Row():
        for text, label in EXAMPLES:
            ex_btn = gr.Button(label, variant="secondary", size="sm")
            ex_btn.click(fn=lambda t=text: t, inputs=[], outputs=[input_text])

    btn.click(fn=correct, inputs=[input_text, beams, use_rag], outputs=[output_text, rag_output])
    input_text.submit(fn=correct, inputs=[input_text, beams, use_rag], outputs=[output_text, rag_output])
    clear_btn.click(fn=lambda: "", inputs=[], outputs=[input_text])

    gr.HTML('<div style="text-align:center;padding:20px;color:#475569;font-size:0.8rem">mT5-small + LoRA + FAISS RAG</div>')

app.launch(server_port=7862, inbrowser=True)
