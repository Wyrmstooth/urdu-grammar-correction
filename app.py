# -*- coding: utf-8 -*-
"""
Urdu Grammar Correction - Web UI
Run: python app.py
Opens at http://127.0.0.1:7860
"""
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Load Model ───
MODEL_PATH = "./urdu_gec_model/final"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("Model loaded!")

# ─── Correction Function ───
def correct_urdu(text, num_beams=4):
    if not text or not text.strip():
        return "براہ کرم اردو جملہ درج کریں"
    input_text = "correct grammar: " + text.strip()
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─── Examples ───
examples = [
    ["میں کل بازار گیا تھا سبزیاں خریدنے کے لیے"],
    ["وہ لڑکی بہت اچھا گاتی ہیں"],
    ["اس نے مجے فون کیا"],
    ["بچے باہر کھیل رہے ہے"],
    ["آج کا کھانا بہت لذیذ ہوئی"],
    ["وہ گھر گیا اور اور پھر واپس آیا"],
    ["کل رات خوب بارس ہوئی"],
    ["بچوں نے شور مچایا بہت کمرے میں"],
    ["وہ ہر روز صبح جوگنگ کرتہ ہے"],
    ["اس نے مجھ کو کچھ نہیں بتایا"],
]

# ─── UI ───
with gr.Blocks(title="Urdu Grammar Correction") as app:
    gr.Markdown("""
    # 📝 اردو گرامر درستی | Urdu Grammar Correction
    
    اپنا اردو جملہ درج کریں اور ماڈل خودکار طریقے سے گرامر کی غلطیاں درست کرے گا۔
    
    *Enter an Urdu sentence and the AI will automatically correct grammar errors.*
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✍️ **ان پٹ / Input**")
            input_text = gr.Textbox(
                label="غلط جملہ (Incorrect Sentence)",
                placeholder="یہاں اپنا اردو جملہ لکھیں...",
                lines=3,
                elem_classes="input-text",
            )
            beams = gr.Slider(1, 8, value=4, step=1, label="Beam Search (زیادہ = بہتر لیکن آہستہ)")
            btn = gr.Button("🔍 گرامر درست کریں | Correct Grammar", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### ✅ **آؤٹ پٹ / Output**")
            output_text = gr.Textbox(
                label="درست جملہ (Corrected Sentence)",
                lines=3,
                interactive=False,
                elem_classes="output-text",
            )

    btn.click(fn=correct_urdu, inputs=[input_text, beams], outputs=output_text)
    input_text.submit(fn=correct_urdu, inputs=[input_text, beams], outputs=output_text)

    gr.Markdown("### 📋 مثالوں سے آزمائیں | Try Examples")
    gr.Examples(examples=examples, inputs=input_text, fn=correct_urdu, outputs=output_text, cache_examples=False)

    gr.Markdown("""
    ---
    ### ℹ️ بارے میں | About
    
    یہ ماڈل **mT5-small** کو 8,143 اردو جملوں پر تربیت دے کر بنایا گیا ہے۔  
    یہ درج ذیل غلطیاں درست کر سکتا ہے:
    - 🏷️ لفظوں کی ترتیب (Word Order)
    - 📐 گرامر (Grammar - verb gender, plural, subject-verb agreement)  
    - ✏️ املا (Spelling)
    - ➕ فالتو الفاظ (Extra Words)
    - ➖ غائب الفاظ (Missing Words)
    
    **BLEU Score: 81%** | **mT5-small (300M)** | **RTX 5060 GPU**
    """)

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
        .output-text textarea { direction: rtl; font-size: 22px !important; }
        .input-text textarea { direction: rtl; font-size: 22px !important; }
        """
    )
