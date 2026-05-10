# -*- coding: utf-8 -*-
"""
Urdu Grammar Rules Knowledge Base for RAG - Cleaned & Corrected
"""
import json

GRAMMAR_RULES = [
    # === Word Order Rules ===
    {
        "id": 1, "category": "word_order", "error_type": "word_order",
        "rule_ur": "فعل جملے کے آخر میں آنا چاہیے",
        "rule_en": "Verb should come at the end of the sentence (SOV order)",
        "example_wrong": "گیا میں بازار",
        "example_right": "میں بازار گیا",
        "explanation": "Urdu follows Subject-Object-Verb order. The verb must come at the end."
    },
    {
        "id": 2, "category": "word_order", "error_type": "word_order",
        "rule_ur": "وقت کا اظہار فاعل کے بعد",
        "rule_en": "Time expressions come after the subject",
        "example_wrong": "کل میں بازار گیا",
        "example_right": "میں کل بازار گیا",
        "explanation": "Time words like 'کل' (yesterday/tomorrow) should come after the subject."
    },
    {
        "id": 3, "category": "word_order", "error_type": "word_order",
        "rule_ur": "صفت اسم سے پہلے",
        "rule_en": "Adjectives come before the noun",
        "example_wrong": "کتاب ایک دلچسپ",
        "example_right": "ایک دلچسپ کتاب",
        "explanation": "Adjectives precede nouns in Urdu. 'اچھی کتاب' (good book), not 'کتاب اچھی'."
    },

    # === Verb Agreement (Gender/Number) ===
    {
        "id": 4, "category": "grammar", "error_type": "verb_agreement",
        "rule_ur": "مفرد مؤنث فاعل کے ساتھ مفرد مؤنث فعل",
        "rule_en": "Singular feminine subject needs singular feminine verb",
        "example_wrong": "وہ لڑکی گاتی ہیں",
        "example_right": "وہ لڑکی گاتی ہے",
        "explanation": "When subject is singular feminine (لڑکی), use singular feminine verb ending (ہے not ہیں)."
    },
    {
        "id": 5, "category": "grammar", "error_type": "verb_agreement",
        "rule_ur": "جمع فاعل کے ساتھ جمع فعل",
        "rule_en": "Plural subjects need plural verb forms",
        "example_wrong": "بچے کھیل رہا ہے",
        "example_right": "بچے کھیل رہے ہیں",
        "explanation": "Plural subjects like 'بچے' need plural verbs. 'رہے ہیں' not 'رہا ہے'."
    },
    {
        "id": 6, "category": "grammar", "error_type": "verb_agreement",
        "rule_ur": "مذکر فاعل کے ساتھ مذکر فعل",
        "rule_en": "Masculine subject needs masculine verb (تھا not تھی)",
        "example_wrong": "استاد نے سبق سمجھایا تھے",
        "example_right": "استاد نے سبق سمجھایا تھا",
        "explanation": "Singular masculine subjects like 'استاد' need masculine singular auxiliary (تھا not تھے)."
    },
    {
        "id": 7, "category": "grammar", "error_type": "verb_agreement",
        "rule_ur": "'ہم' کے ساتھ جمع مذکر فعل",
        "rule_en": "The pronoun 'ہم' (we) takes masculine plural verbs",
        "example_wrong": "ہم بازار گئے تھا",
        "example_right": "ہم بازار گئے تھے",
        "explanation": "'ہم' (we) always requires plural verb forms. 'گئے تھے' not 'گئے تھا'."
    },
    {
        "id": 8, "category": "grammar", "error_type": "verb_agreement",
        "rule_ur": "مؤنث فاعل کے ساتھ مؤنث فعل (ماضی)",
        "rule_en": "Feminine subject needs feminine past verb (گئی not گیا)",
        "example_wrong": "میری بہن لاہور گیا تھی",
        "example_right": "میری بہن لاہور گئی تھی",
        "explanation": "'بہن' (sister) is feminine. The past verb must be feminine: 'گئی تھی' not 'گیا تھی'."
    },
    {
        "id": 9, "category": "grammar", "error_type": "verb_agreement",
        "rule_ur": "جمع فاعل کے ساتھ جمع فعل (ماضی)",
        "rule_en": "Plural subjects need plural verbs throughout the sentence",
        "example_wrong": "بچے پارک میں کھیل رہا تھا",
        "example_right": "بچے پارک میں کھیل رہے تھے",
        "explanation": "'بچے' is plural. Both 'رہا' and 'تھا' must become plural: 'رہے تھے'."
    },

    # === Ergative Constructions (نے) ===
    {
        "id": 10, "category": "grammar", "error_type": "ergative",
        "rule_ur": "نے والے جملوں میں فعل مفعول سے مطابقت رکھتا ہے",
        "rule_en": "After 'نے', the verb agrees with the object, not the subject",
        "example_wrong": "پولیس نے رپورٹ درج کیا",
        "example_right": "پولیس نے رپورٹ درج کی",
        "explanation": "'رپورٹ' is feminine. After 'نے', use feminine verb 'کی' not masculine 'کیا'."
    },
    {
        "id": 11, "category": "grammar", "error_type": "ergative",
        "rule_ur": "نے کے بعد مذکر مفعول کے ساتھ مذکر فعل",
        "rule_en": "After 'نے' with masculine object, use masculine singular verb",
        "example_wrong": "حکومت نے اعلان کئے",
        "example_right": "حکومت نے اعلان کیا",
        "explanation": "'اعلان' is masculine singular. Use 'کیا' not plural 'کئے'."
    },
    {
        "id": 12, "category": "grammar", "error_type": "ergative",
        "rule_ur": "مؤنث مفعول کے ساتھ مؤنث فعل (نے)",
        "rule_en": "Feminine object with نے needs feminine verb",
        "example_wrong": "حکومت نے سہولتیں شروع کیا ہیں",
        "example_right": "حکومت نے سہولتیں شروع کی ہیں",
        "explanation": "'سہولتیں' is feminine. The verb must be feminine 'کی' not 'کیا'."
    },
    {
        "id": 13, "category": "grammar", "error_type": "ergative",
        "rule_ur": "میں نے کے بعد مؤنث مفعول",
        "rule_en": "'میں نے' with feminine object needs feminine verb",
        "example_wrong": "میں نے نوکری شروع کیا ہے",
        "example_right": "میں نے نوکری شروع کی ہے",
        "explanation": "'نوکری' is feminine. Use 'کی' not 'کیا'. Similarly: میں نے کتاب پڑھی. I"
    },

    # === Adjective-Noun Agreement ===
    {
        "id": 14, "category": "grammar", "error_type": "adjective_agreement",
        "rule_ur": "صفت اور اسم کی جنس میں مطابقت",
        "rule_en": "Adjective must match the noun's gender",
        "example_wrong": "یہ کھانا بہت اچھے ہے",
        "example_right": "یہ کھانا بہت اچھا ہے",
        "explanation": "'کھانا' is masculine. The adjective must be masculine 'اچھا' not 'اچھے'."
    },
    {
        "id": 15, "category": "grammar", "error_type": "adjective_agreement",
        "rule_ur": "جمع مؤنث اسم کے ساتھ جمع مؤنث فعل",
        "rule_en": "Feminine plural nouns need feminine plural verbs",
        "example_wrong": "یہ تصویریں خوبصورت تھا",
        "example_right": "یہ تصویریں خوبصورت تھیں",
        "explanation": "'تصویریں' is feminine plural. Use 'تھیں' not 'تھا'."
    },

    # === Auxiliary Verb Agreement ===
    {
        "id": 16, "category": "grammar", "error_type": "auxiliary",
        "rule_ur": "فعل معاون کی جنس فاعل کے مطابق",
        "rule_en": "Auxiliary verb must agree with subject in gender and number",
        "example_wrong": "وہ موبائل پر گانے سن رہی تھا",
        "example_right": "وہ موبائل پر گانے سن رہی تھی",
        "explanation": "The main verb 'سن رہی' is feminine. Auxiliary must match: 'تھی' not 'تھا'."
    },
    {
        "id": 17, "category": "grammar", "error_type": "auxiliary",
        "rule_ur": "فاعل واحد کے ساتھ 'ہے' اور جمع کے ساتھ 'ہیں'",
        "rule_en": "Singular subject uses 'ہے', plural uses 'ہیں'",
        "example_wrong": "وہ اسکول دیر سے پہنچتی ہیں",
        "example_right": "وہ اسکول دیر سے پہنچتی ہے",
        "explanation": "Singular feminine subject uses 'ہے' not 'ہیں'. 'پہنچتی ہے' is correct."
    },
    {
        "id": 18, "category": "grammar", "error_type": "auxiliary",
        "rule_ur": "پہلا شخص 'میں' کے ساتھ 'ہوں'",
        "rule_en": "First person 'میں' uses 'ہوں' or appropriate auxiliary",
        "example_wrong": "میں نے ناشتہ نہیں کیا ہیں",
        "example_right": "میں نے ناشتہ نہیں کیا ہے",
        "explanation": "'میں نے' takes singular auxiliary 'ہے' not plural 'ہیں'."
    },

    # === Natural Phenomena ===
    {
        "id": 19, "category": "grammar", "error_type": "gender_agreement",
        "rule_ur": "بارش اور قدرتی مظاہر مؤنث ہیں",
        "rule_en": "Rain, snow, and weather phenomena are feminine",
        "example_wrong": "بارش سے سڑک بند ہو گیا",
        "example_right": "بارش سے سڑک بند ہو گئی",
        "explanation": "'بارش' (rain) is feminine. Use feminine verb 'ہو گئی' not 'ہو گیا'."
    },
    {
        "id": 20, "category": "grammar", "error_type": "gender_agreement",
        "rule_ur": "برف باری مؤنث ہے",
        "rule_en": "Snowfall (برف باری) is feminine singular",
        "example_wrong": "مری میں برف باری ہو رہے ہیں",
        "example_right": "مری میں برف باری ہو رہی ہے",
        "explanation": "'برف باری' is feminine singular. Use 'ہو رہی ہے' not 'ہو رہے ہیں'."
    },

    # === Spelling / Typing ===
    {
        "id": 21, "category": "spelling", "error_type": "typo",
        "rule_ur": "مجھے کی درست املا",
        "rule_en": "Correct spelling: مجھے (mujhe) not مجے",
        "example_wrong": "اس نے مجے فون کیا",
        "example_right": "اس نے مجھے فون کیا",
        "explanation": "'مجے' is missing the 'ھ' (do-chashmi he). The correct spelling is 'مجھے'."
    },
    {
        "id": 22, "category": "spelling", "error_type": "typo",
        "rule_ur": "بارش کی درست املا",
        "rule_en": "Correct spelling: بارش (baarish) not بارس",
        "example_wrong": "کل رات خوب بارس ہوئی",
        "example_right": "کل رات خوب بارش ہوئی",
        "explanation": "'بارش' (rain) has 'ش' (sheen), not 'س' (seen)."
    },
    {
        "id": 23, "category": "spelling", "error_type": "typo",
        "rule_ur": "جوگنگ کی درست املا",
        "rule_en": "Correct: جوگنگ (jogging) not جانچ",
        "example_wrong": "وہ ہر روز صبح جانچ کرتہ ہے",
        "example_right": "وہ ہر روز صبح جوگنگ کرتا ہے",
        "explanation": "'جانچ' means checking. The English word 'jogging' is written as 'جوگنگ' in Urdu."
    },
    {
        "id": 24, "category": "spelling", "error_type": "typo",
        "rule_ur": "چلی گئی کی درست املا",
        "rule_en": "Complete the word: گئی not گئ",
        "example_wrong": "کل رات بجلی چلی گئ",
        "example_right": "کل رات بجلی چلی گئی",
        "explanation": "Don't omit the final 'ی' (ye). 'گئی' is the complete feminine past form."
    },

    # === Extra Words ===
    {
        "id": 25, "category": "extra_words", "error_type": "redundancy",
        "rule_ur": "مکرر الفاظ کو حذف کریں",
        "rule_en": "Remove duplicate words",
        "example_wrong": "وہ گھر گیا اور اور پھر واپس آیا",
        "example_right": "وہ گھر گیا اور پھر واپس آیا",
        "explanation": "Don't repeat 'اور' (and). One 'اور' is enough to connect two clauses."
    },
    {
        "id": 26, "category": "extra_words", "error_type": "redundancy",
        "rule_ur": "فالتو شدت نما الفاظ",
        "rule_en": "Remove redundant intensifiers",
        "example_wrong": "بہت زیادہ بہت اچھا",
        "example_right": "بہت اچھا",
        "explanation": "Don't stack intensifiers like 'بہت زیادہ بہت'. Use one intensifier."
    },

    # === Missing Words ===
    {
        "id": 27, "category": "missing_words", "error_type": "case_marker",
        "rule_ur": "مقام کے لیے 'میں' کا استعمال",
        "rule_en": "Add 'میں' for location",
        "example_wrong": "میں نے بازار سبزی خریدی",
        "example_right": "میں نے بازار میں سبزی خریدی",
        "explanation": "Location nouns need 'میں' (in). 'بازار میں' = 'in the market'."
    },
    {
        "id": 28, "category": "missing_words", "error_type": "case_marker",
        "rule_ur": "'کو' کا استعمال مفعول کے لیے",
        "rule_en": "Add 'کو' for definite direct objects",
        "example_wrong": "میں نے لڑکی دیکھا",
        "example_right": "میں نے لڑکی کو دیکھا",
        "explanation": "Definite animate objects need 'کو'. 'لڑکی کو' not just 'لڑکی'."
    },

    # === Mixed Patterns ===
    {
        "id": 29, "category": "grammar", "error_type": "mixed",
        "rule_ur": "'وہ' کے بعد فعل کی مختلف صورتیں",
        "rule_en": "'وہ' can be singular or plural - verb must match context",
        "example_wrong": "وہ دوست سے ملنے آئے تھی",
        "example_right": "وہ دوست سے ملنے آئی تھی",
        "explanation": "When 'وہ' refers to a single female, use feminine verbs: 'آئی تھی' not 'آئے تھی'."
    },
    {
        "id": 30, "category": "grammar", "error_type": "mixed",
        "rule_ur": "شور مچانا - فعل کی مطابقت",
        "rule_en": "Compound verbs: agreement follows the subject",
        "example_wrong": "بچے کلاس میں شور مچا رہا تھا",
        "example_right": "بچے کلاس میں شور مچا رہے تھے",
        "explanation": "'بچے' is plural. The auxiliary system must be plural: 'رہے تھے' not 'رہا تھا'."
    },
]

if __name__ == "__main__":
    with open("urdu_grammar_rules.json", "w", encoding="utf-8") as f:
        json.dump(GRAMMAR_RULES, f, ensure_ascii=False, indent=2)
    print(f"Written {len(GRAMMAR_RULES)} rules to urdu_grammar_rules.json")
