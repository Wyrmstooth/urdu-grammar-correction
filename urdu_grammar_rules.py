# -*- coding: utf-8 -*-
"""
Urdu Grammar Rules Knowledge Base for RAG
Each entry: rule text (Urdu), category, error type, explanation (English)
"""
import json

GRAMMAR_RULES = [
    # Word Order Rules
    {
        "id": 1,
        "category": "word_order",
        "error_type": "word_order",
        "rule_ur": "فاعل کا فعل سے پہلے آنا چاہیے",
        "rule_en": "Subject should come before verb",
        "example_wrong": "گیا میں بازار",
        "example_right": "میں بازار گیا",
        "explanation": "In Urdu, the standard sentence structure is Subject-Object-Verb (SOV). The verb usually comes at the end of the sentence."
    },
    {
        "id": 2,
        "category": "word_order",
        "error_type": "word_order",
        "rule_ur": "مفعول کا فاعل کے بعد آنا چاہیے",
        "rule_en": "Object should come after subject",
        "example_wrong": "بازار میں گیا میں",
        "example_right": "میں بازار میں گیا",
        "explanation": "After subject, the object/location phrase follows before the verb."
    },
    {
        "id": 3,
        "category": "word_order",
        "error_type": "word_order",
        "rule_ur": "قصدیہ جملے میں صفت کا بعد میں آنا",
        "rule_en": "Adverbial phrases should be placed correctly",
        "example_wrong": "کل میں بازار گیا",
        "example_right": "میں کل بازار گیا",
        "explanation": "Time expressions like 'کل' (yesterday/tomorrow) typically come early in the sentence but after the subject."
    },
    # Verb Agreement Rules
    {
        "id": 4,
        "category": "grammar",
        "error_type": "verb_agreement",
        "rule_ur": "singular feminine subjects take singular feminine verbs",
        "rule_en": "Singular feminine subjects need singular feminine verb forms",
        "example_wrong": "وہ لڑکی گاتی ہیں",
        "example_right": "وہ لڑکی گاتی ہے",
        "explanation": "When the subject is singular feminine (لڑکی), the verb must also be singular feminine (ہے not ہیں)."
    },
    {
        "id": 5,
        "category": "grammar",
        "error_type": "verb_agreement",
        "rule_ur": "plural subjects take plural verbs",
        "rule_en": "Plural subjects require plural verb forms",
        "example_wrong": "بچے کھیل رہا ہے",
        "example_right": "بچے کھیل رہے ہیں",
        "explanation": "Plural subjects like 'بچے' (children) require plural verbs like 'ہیں' not singular 'ہے'."
    },
    {
        "id": 6,
        "category": "grammar",
        "error_type": "verb_agreement",
        "rule_ur": "masculine plural subjects need masculine plural verbs",
        "rule_en": "Masculine plural subjects need masculine plural verb agreement",
        "example_wrong": "وہ لوگ آتا ہے",
        "example_right": "وہ لوگ آتے ہیں",
        "explanation": "For masculine plural subjects, use masculine plural verb forms like 'آتے ہیں' not 'آتا ہے'."
    },
    # Case Markers (آ)
    {
        "id": 7,
        "category": "missing_words",
        "error_type": "case_marker",
        "rule_ur": "کو کا استعمال",
        "rule_en": "Use 'کو' for indirect object",
        "example_wrong": "اس نے مجے فون کیا",
        "example_right": "اس نے مجھے فون کیا",
        "explanation": "'مجے' is incorrect. The correct form is 'مجھے' (mujhe) which is the indirect object marker 'کو' attached to 'مجھ'."
    },
    {
        "id": 8,
        "category": "missing_words",
        "error_type": "case_marker",
        "rule_ur": "میں کا استعمال برائے مقام",
        "rule_en": "Use 'میں' for location",
        "example_wrong": "بازار سبزی خریدی",
        "example_right": "میں بازار میں سبزی خریدی",
        "explanation": "Location words need 'میں' (in). 'بازار میں' means 'in the market'. Without it, the sentence is incomplete."
    },
    {
        "id": 9,
        "category": "missing_words",
        "error_type": "case_marker",
        "rule_ur": "نے کا استعمال فاعل کے ساتھ",
        "rule_en": "Use 'نے' with transitive verb subjects",
        "example_wrong": "اس نے کھایا کھانا",
        "example_right": "اس نے کھانا کھایا",
        "explanation": "The subject of a transitive verb in past tense takes 'نے'. 'اس نے' means 'he/she did' with an agent marker."
    },
    # Extra Words
    {
        "id": 10,
        "category": "extra_words",
        "error_type": "redundancy",
        "rule_ur": "اور کا duplaicate elimination",
        "rule_en": "Remove duplicate 'اور' (and)",
        "example_wrong": "وہ گیا اور اور واپس آیا",
        "example_right": "وہ گیا اور واپس آیا",
        "explanation": "When using 'اور' (and) to connect two actions, do not repeat it. One 'اور' is sufficient."
    },
    {
        "id": 11,
        "category": "extra_words",
        "error_type": "redundancy",
        "rule_ur": "بہت کا overstated usage",
        "rule_en": "Avoid redundant modifiers",
        "example_wrong": "بہت زیادہ بہت زیادہ",
        "example_right": "بہت زیادہ",
        "explanation": "Do not repeat intensifiers like 'بہت' or 'زیادہ'. Using once is sufficient."
    },
    # Spelling Rules
    {
        "id": 12,
        "category": "spelling",
        "error_type": "typo",
        "rule_ur": "مجے → مجھے",
        "rule_en": "Correct spelling: مجھے (mujhe) not مجے",
        "example_wrong": "اس نے مجے فون کیا",
        "example_right": "اس نے مجھے فون کیا",
        "explanation": "'مجے' is a common misspelling. The correct form is 'مجھے' with the 'ھ' (h) character."
    },
    {
        "id": 13,
        "category": "spelling",
        "error_type": "typo",
        "rule_ur": "بارس → بارش",
        "rule_en": "Correct spelling: بارش (rain) not بارس",
        "example_wrong": "کل رات خوب بارس ہوئی",
        "example_right": "کل رات خوب بارش ہوئی",
        "explanation": "'بارس' (baaris) is wrong. 'بارش' (baarish) means rain. The 'ش' (sh) is essential."
    },
    {
        "id": 14,
        "category": "spelling",
        "error_type": "typo",
        "rule_ur": "کمرے → کمرے",
        "rule_en": "Room is کمرے not کمرے",
        "example_wrong": "بچوں نے شور مچایا بہت کمرے میں",
        "example_right": "بچوں نے شور مچایا بہت کمرے میں",
        "explanation": "'کمرے' means room. The spelling involves 'ک' + 'م' + 'ر' + 'ے' and is a Persian-derived word."
    },
    {
        "id": 15,
        "category": "spelling",
        "error_type": "typo",
        "rule_ur": "لذیذ → لذیذ",
        "rule_en": "Delicious is لذیذ not لذیذ",
        "example_wrong": "آج کا کھانا بہت لذیذ ہوئی",
        "example_right": "آج کا کھانا بہت لذیذ ہوا",
        "explanation": "'لذیذ' vs 'لذیذ' - both exist but 'لذیذ' (laziz) is more commonly used for tasty. Also 'ہوئی' should be 'ہوا' for masculine."
    },
    # Gender Agreement
    {
        "id": 16,
        "category": "grammar",
        "error_type": "gender_agreement",
        "rule_ur": "مؤنث فاعل → مؤنث فعل",
        "rule_en": "Feminine subject needs feminine verb",
        "example_wrong": "وہ عورت کام کرائی",
        "example_right": "وہ عورت کام کراؤتی ہے",
        "explanation": "Feminine subjects require feminine verb forms. 'کرائی' is wrong. Use 'کراؤتی ہے' for ongoing feminine action."
    },
    {
        "id": 17,
        "category": "grammar",
        "error_type": "gender_agreement",
        "rule_ur": "مذکر جمع فاعل → مذکر جمع فعل",
        "rule_en": "Masculine plural subject needs masculine plural verb",
        "example_wrong": "لوگوں نے کھایا",
        "example_right": "لوگوں نے کھایا",
        "explanation": "'لوگوں' (people) is masculine plural. For past tense with 'نے', the verb form 'کھایا' agrees with masculine singular. For plural, use context."
    },
    # Tense Consistency
    {
        "id": 18,
        "category": "grammar",
        "error_type": "tense",
        "rule_ur": "ماضی بعید کا استعمال",
        "rule_en": "Use past perfect for actions before another past action",
        "example_wrong": "وہ آیا تھا جب میں نے پکڑا",
        "example_right": "وہ آگیا تھا جب میں نے پکڑا",
        "explanation": "When an action happened before another past action, use perfect aspect. 'آگیا تھا' (had come) vs 'آیا تھا' (was coming)."
    },
    {
        "id": 19,
        "category": "grammar",
        "error_type": "tense",
        "rule_ur": "حال میں continueing action",
        "rule_en": "Use present continuous for ongoing actions",
        "example_wrong": "وہ کھیلنا",
        "example_right": "وہ کھیل رہا ہے",
        "explanation": "For ongoing actions in present, use 'رہا ہے' form. 'کھیل رہا ہے' means 'is playing'."
    },
    # Common Errors
    {
        "id": 20,
        "category": "spelling",
        "error_type": "common",
        "rule_ur": "ہے ← ہے",
        "rule_en": "is verb should always have ہے not confused with ہی",
        "example_wrong": "وہ گھر ہے گیا",
        "example_right": "وہ گھر گیا",
        "explanation": "The verb 'ہے' (is) should not be combined with another verb form incorrectly. Use simple past or appropriate compound form."
    },
    {
        "id": 21,
        "category": "word_order",
        "error_type": "question",
        "rule_ur": "سوالیہ جملے کی ساخت",
        "rule_en": "Question sentence structure in Urdu",
        "example_wrong": "کیا تم آئے ہو؟",
        "example_right": "کیا آئے ہو؟",
        "explanation": "In questions, 'کیا' starts the question but the subject can come after. 'کیا تم آئے ہو؟' is formal, 'کیا آئے ہو؟' is colloquial."
    },
    {
        "id": 22,
        "category": "missing_words",
        "error_type": "preposition",
        "rule_ur": "سے کا استعمال",
        "rule_en": "Use 'سے' for 'from/with/by'",
        "example_wrong": "میں نے اس سے بات کریے",
        "example_right": "میں نے اس سے بات کی",
        "explanation": "'بات کریے' is wrong. 'بات کرنا' (to talk) needs correct conjugation. 'بات کی' is past tense of feminine noun."
    },
    {
        "id": 23,
        "category": "grammar",
        "error_type": "conjunction",
        "rule_ur": "جو کا استعمال",
        "rule_en": "Use 'جو' for relative clauses",
        "example_wrong": "وہ آدمی جو میں نے دیکھا سے",
        "example_right": "وہ آدمی جسے میں نے دیکھا",
        "explanation": "'جو' (who) has different forms: 'جو' for subject, 'جسے' for indirect object, 'جس نے' for agent. Choose based on grammatical role."
    },
    {
        "id": 24,
        "category": "extra_words",
        "error_type": "filler",
        "rule_ur": "فالتو الفاظ کی شناخت",
        "rule_en": "Identify and remove filler words",
        "example_wrong": "وہ تو وہی تو ہے",
        "example_right": "وہ وہی ہے",
        "explanation": "Filler words like repeated 'تو' should be removed. 'وہ تو وہی تو ہے' becomes 'وہ وہی ہے' (It is the same one)."
    },
    {
        "id": 25,
        "category": "spelling",
        "error_type": "typo",
        "rule_ur": "جوگنگ ← جانچ",
        "rule_en": "Jogging is جوگنگ not جانچ",
        "example_wrong": "وہ ہر روز صبح جانچ کرتہ ہے",
        "example_right": "وہ ہر روز صبح جوگنگ کرتا ہے",
        "explanation": "'جانچ' means 'checking/testing'. 'جوگنگ' (jogging) is the correct Urdu transliteration of the English word."
    },
    {
        "id": 26,
        "category": "missing_words",
        "error_type": "article",
        "rule_ur": "ایک کا استعمال",
        "rule_en": "Use 'ایک' (one/a) as indefinite article",
        "example_wrong": "میں نے کتاب پڑھی",
        "example_right": "میں نے ایک کتاب پڑھی",
        "explanation": "For indefinite objects, use 'ایک' before the noun. 'ایک کتاب' means 'a book'. This is often omitted in colloquial speech but preferred in formal writing."
    },
    {
        "id": 27,
        "category": "grammar",
        "error_type": "number",
        "rule_ur": "جمع کی گرامر",
        "rule_en": "Plural formation in Urdu",
        "example_wrong": "بچے باہر کھیل رہے ہے",
        "example_right": "بچے باہر کھیل رہے ہیں",
        "explanation": "'بچے' is already plural. The verb must also be plural 'ہیں' not singular 'ہے'."
    },
    {
        "id": 28,
        "category": "missing_words",
        "error_type": "possessive",
        "rule_ur": "کا/کی/کے کا استعمال",
        "rule_en": "Use possessive markers کا/کی/کے correctly",
        "example_wrong": "میں گھر جا رہا ہوں",
        "example_right": "میں گھر جا رہا ہوں",
        "explanation": "Possessive 'کا/کی/کے' must agree in gender with the possessed noun. 'گھر' (home) is masculine, so no marker needed or use 'کا' if followed."
    },
    {
        "id": 29,
        "category": "spelling",
        "error_type": "typo",
        "rule_ur": "بتایا ← بتایا",
        "rule_en": "Told is بتایا not بتایا",
        "example_wrong": "اس نے مجھ کو کچھ نہیں بتایا",
        "example_right": "اس نے مجھے کچھ نہیں بتایا",
        "explanation": "'بتایا' (told) is the correct form. The common error is with the silent 'ے' at the end."
    },
    {
        "id": 30,
        "category": "word_order",
        "error_type": "adjective",
        "rule_ur": "صفت کی پوزیشن",
        "rule_en": "Adjective should come before noun",
        "example_wrong": "کتاب ایک مزید دلچسپ",
        "example_right": "ایک مزید دلچسپ کتاب",
        "explanation": "In Urdu, adjectives precede the noun they modify. 'دلچسپ کتاب' (interesting book) not 'کتاب دلچسپ'."
    }
]


def save_knowledge_base():
    with open("urdu_grammar_rules.json", "w", encoding="utf-8") as f:
        json.dump(GRAMMAR_RULES, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(GRAMMAR_RULES)} grammar rules to urdu_grammar_rules.json")


if __name__ == "__main__":
    save_knowledge_base()
