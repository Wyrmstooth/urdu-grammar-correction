# -*- coding: utf-8 -*-
"""Bulk grammar data generator - 2000+ pairs covering gender/number agreement"""
import json, random
random.seed(42)

# All verb forms by gender/number
# (corrupted_masc_sing, corrupted_fem_sing, corrupted_masc_plur, corrupted_fem_plur)
# The correct form depends on subject

# Key: correct form, Value: list of [wrong_masc_sing, wrong_fem_sing, wrong_masc_plur, wrong_fem_plur]
VERB_VARIANTS = {
    # Past tense main verbs
    "گیا":  ["گیا",  "گئی",  "گئے",  "گئیں"],
    "گئی":  ["گیا",  "گئی",  "گئے",  "گئیں"],
    "گئے":  ["گیا",  "گئی",  "گئے",  "گئیں"],
    "گئیں": ["گیا",  "گئی",  "گئے",  "گئیں"],
    # Past auxiliaries
    "تھا":  ["تھا",  "تھی",  "تھے",  "تھیں"],
    "تھی":  ["تھا",  "تھی",  "تھے",  "تھیں"],
    "تھے":  ["تھا",  "تھی",  "تھے",  "تھیں"],
    "تھیں": ["تھا",  "تھی",  "تھے",  "تھیں"],
    # Present auxiliaries  
    "ہے":   ["ہے",   "ہے",   "ہیں",  "ہیں"],
    "ہیں":  ["ہے",   "ہے",   "ہیں",  "ہیں"],
    # Present/habitual verbs
    "جاتا":  ["جاتا",  "جاتی",  "جاتے",  "جاتیں"],
    "جاتی":  ["جاتا",  "جاتی",  "جاتے",  "جاتیں"],
    "جاتے":  ["جاتا",  "جاتی",  "جاتے",  "جاتیں"],
    "جاتیں": ["جاتا",  "جاتی",  "جاتے",  "جاتیں"],
    "کرتا":  ["کرتا",  "کرتی",  "کرتے",  "کرتیں"],
    "کرتی":  ["کرتا",  "کرتی",  "کرتے",  "کرتیں"],
    "کرتے":  ["کرتا",  "کرتی",  "کرتے",  "کرتیں"],
    "کرتیں": ["کرتا",  "کرتی",  "کرتے",  "کرتیں"],
    "آتا":   ["آتا",   "آتی",   "آتے",   "آتیں"],
    "آتی":   ["آتا",   "آتی",   "آتے",   "آتیں"],
    "آتے":   ["آتا",   "آتی",   "آتے",   "آتیں"],
    "آتیں":  ["آتا",   "آتی",   "آتے",   "آتیں"],
    "رہتا":  ["رہتا",  "رہتی",  "رہتے",  "رہتیں"],
    "رہتی":  ["رہتا",  "رہتی",  "رہتے",  "رہتیں"],
    # Continuous aspect
    "رہا":   ["رہا",   "رہی",   "رہے",   "رہیں"],
    "رہی":   ["رہا",   "رہی",   "رہے",   "رہیں"],
    "رہے":   ["رہا",   "رہی",   "رہے",   "رہیں"],
    "رہیں":  ["رہا",   "رہی",   "رہے",   "رہیں"],
    # Perfect forms
    "کیا":   ["کیا",   "کی",    "کئے",   "کیں"],
    "کی":    ["کیا",   "کی",    "کئے",   "کیں"],
    "کئے":   ["کیا",   "کی",    "کئے",   "کیں"],
    "کیں":   ["کیا",   "کی",    "کئے",   "کیں"],
    "دیا":   ["دیا",   "دی",    "دئے",   "دیں"],
    "دی":    ["دیا",   "دی",    "دئے",   "دیں"],
    "دئے":   ["دیا",   "دی",    "دئے",   "دیں"],
    "لیا":   ["لیا",   "لی",    "لئے",   "لیں"],
    "لی":    ["لیا",   "لی",    "لئے",   "لیں"],
    "لئے":   ["لیا",   "لی",    "لئے",   "لیں"],
    "ہوا":   ["ہوا",   "ہوئی",  "ہوئے",  "ہوئیں"],
    "ہوئی":  ["ہوا",   "ہوئی",  "ہوئے",  "ہوئیں"],
    "ہوئے":  ["ہوا",   "ہوئی",  "ہوئے",  "ہوئیں"],
    "ہوئیں": ["ہوا",   "ہوئی",  "ہوئے",  "ہوئیں"],
    # First person
    "ہوں":   ["ہوں",   "ہوں",   "ہیں",   "ہیں"],
    # Adjectives
    "اچھا":  ["اچھا",  "اچھی",  "اچھے",  "اچھی"],
    "اچھی":  ["اچھا",  "اچھی",  "اچھے",  "اچھی"],
    "اچھے":  ["اچھا",  "اچھی",  "اچھے",  "اچھی"],
    "خوبصورت": ["خوبصورت"] * 4,  # invariable
    "بڑا":   ["بڑا",   "بڑی",   "بڑے",   "بڑی"],
    "بڑی":   ["بڑا",   "بڑی",   "بڑے",   "بڑی"],
    "بڑے":   ["بڑا",   "بڑی",   "بڑے",   "بڑی"],
    "لمبا":  ["لمبا",  "لمبی",  "لمبے",  "لمبی"],
    "لمبی":  ["لمبا",  "لمبی",  "لمبے",  "لمبی"],
    "لمبے":  ["لمبا",  "لمبی",  "لمبے",  "لمبی"],
    "ذہین":  ["ذہین"] * 4,  # invariable
    "مشکل":  ["مشکل"] * 4,
    "ٹھنڈا": ["ٹھنڈا", "ٹھنڈی", "ٹھنڈے", "ٹھنڈی"],
    "ٹھنڈی": ["ٹھنڈا", "ٹھنڈی", "ٹھنڈے", "ٹھنڈی"],
    "گرم":   ["گرم"] * 4,
}

GENDER_ORDER = {"masc_sing": 0, "fem_sing": 1, "masc_plur": 2, "fem_plur": 3}

def corrupt(sentence, target_gender_number, word_indices=None):
    """Corrupt a correct sentence to a specific wrong gender/number."""
    words = sentence.split()
    if word_indices is None:
        # Auto-detect: corrupt last 1-2 verbs/auxiliaries
        word_indices = list(range(max(0, len(words)-3), len(words)))
    
    wrong_idx = GENDER_ORDER[target_gender_number]
    corrupted = words[:]
    changed = False
    for idx in word_indices:
        if idx < len(words):
            w = words[idx]
            if w in VERB_VARIANTS:
                variant = VERB_VARIANTS[w][wrong_idx]
                if variant != w:
                    corrupted[idx] = variant
                    changed = True
    
    if changed:
        return " ".join(corrupted)
    return None

# ─── Generate templates: (correct_sentence, correct_gender_number, category) ───
templates = []

# === MASC SINGULAR subjects ===
masc_sing_subjects = [
    "استاد", "ڈاکٹر", "لڑکا", "بچہ", "آدمی", "کسان", "مالی", "نوکر", "وزیر",
    "طلب علم", "طالب علم", "شاگرد", "بیٹا", "باپ", "بھائی", "دوست", "ہمسایہ",
    "انجینئر", "وکیل", "جج", "پولیس والا", "دکاندار", "ملازم", "افسر", "مزدور",
]

masc_sing_actions = [
    ("اسکول گیا تھا", "had gone to school"),
    ("کام کر رہا تھا", "was working"),
    ("کھانا کھا رہا تھا", "was eating"),
    ("سبق پڑھ رہا تھا", "was reading lesson"),
    ("پانی پی رہا تھا", "was drinking water"),
    ("سو رہا تھا", "was sleeping"),
    ("جاگ رہا تھا", "was awake"),
    ("بازار جا رہا تھا", "was going to market"),
    ("کتاب پڑھ رہا تھا", "was reading book"),
    ("اخبار دیکھ رہا تھا", "was reading newspaper"),
    ("ٹی وی دیکھ رہا تھا", "was watching TV"),
    ("گانا گا رہا تھا", "was singing"),
    ("خط لکھ رہا تھا", "was writing letter"),
    ("روٹی کھا رہا تھا", "was eating bread"),
    ("چائے پی رہا تھا", "was drinking tea"),
    ("دفتر جا رہا تھا", "was going to office"),
    ("مسجد جا رہا تھا", "was going to mosque"),
    ("دوڑ رہا تھا", "was running"),
    ("بیٹھا تھا", "was sitting"),
    ("کھڑا تھا", "was standing"),
]

for subj in masc_sing_subjects:
    for action, _ in masc_sing_actions[:5]:  # 5 actions per subject = 125 pairs
        sentence = f"{subj} {action}"
        templates.append((sentence, "masc_sing", "daily"))

# === FEM SINGULAR subjects ===
fem_sing_subjects = [
    "لڑکی", "بچی", "عورت", "استانی", "ماں", "بیٹی", "بہن", "خاتون",
    "طالبہ", "نرس", "ہمسائی", "ملازمہ",
]

fem_sing_actions = [
    ("اسکول گئی تھی", "had gone to school"),
    ("کام کر رہی تھی", "was working"),
    ("کھانا کھا رہی تھی", "was eating"),
    ("سبق پڑھ رہی تھی", "was reading lesson"),
    ("پانی پی رہی تھی", "was drinking water"),
    ("سو رہی تھی", "was sleeping"),
    ("جاگ رہی تھی", "was awake"),
    ("بازار جا رہی تھی", "was going to market"),
    ("کتاب پڑھ رہی تھی", "was reading book"),
    ("گانا گا رہی تھی", "was singing"),
    ("دفتر جا رہی تھی", "was going to office"),
    ("بیٹھی تھی", "was sitting"),
    ("چائے بنا رہی تھی", "was making tea"),
    ("کپڑے دھو رہی تھی", "was washing clothes"),
    ("گھر صاف کر رہی تھی", "was cleaning house"),
]

for subj in fem_sing_subjects:
    for action, _ in fem_sing_actions[:2]:  # 2 actions per = 24 pairs
        sentence = f"{subj} {action}"
        templates.append((sentence, "fem_sing", "daily"))

# === MASC PLURAL subjects ===
masc_plur_subjects = [
    "بچے", "لڑکے", "مزدور", "کسان", "طلبہ", "شاگرد", "ملازم",
    "دوست", "کھلاڑی", "افسر",
]

masc_plur_actions = [
    ("پارک میں کھیل رہے تھے", "were playing in park"),
    ("میدان میں دوڑ رہے تھے", "were running in field"),
    ("کلاس میں شور مچا رہے تھے", "were making noise in class"),
    ("کام کر رہے تھے", "were working"),
    ("کھانا کھا رہے تھے", "were eating"),
    ("کرکٹ کھیل رہے تھے", "were playing cricket"),
    ("باتیں کر رہے تھے", "were talking"),
    ("پڑھ رہے تھے", "were studying"),
    ("سو رہے تھے", "were sleeping"),
]

for subj in masc_plur_subjects:
    for action, _ in masc_plur_actions[:2]:
        sentence = f"{subj} {action}"
        templates.append((sentence, "masc_plur", "daily"))

# === FEM PLURAL subjects ===
fem_plur_subjects = [
    "لڑکیاں", "بچیاں", "عورتیں", "طالبات",
]

fem_plur_actions = [
    ("گانا گا رہی تھیں", "were singing"),
    ("پارک میں کھیل رہی تھیں", "were playing in park"),
    ("باتیں کر رہی تھیں", "were talking"),
    ("کھانا پکا رہی تھیں", "were cooking"),
    ("پڑھ رہی تھیں", "were studying"),
]

for subj in fem_plur_subjects:
    for action, _ in fem_plur_actions[:3]:
        sentence = f"{subj} {action}"
        templates.append((sentence, "fem_plur", "daily"))

# === ہم (we) patterns ===
hum_actions_masc = [
    ("بازار گئے تھے", "had gone to market"),
    ("کام کر رہے تھے", "were working"),
    ("کھانا کھا رہے تھے", "were eating"),
    ("پارک میں کھیل رہے تھے", "were playing"),
    ("رات تک پڑھتے رہے", "kept studying till night"),
    ("سارا دن سوتے رہے", "kept sleeping all day"),
    ("شام تک انتظار کرتے رہے", "kept waiting till evening"),
    ("کام کرتے رہے", "kept working"),
    ("گپ شپ کرتے رہے", "kept chatting"),
]

for action, _ in hum_actions_masc:
    sentence = f"ہم {action}"
    templates.append((sentence, "masc_plur", "daily"))

hum_actions_fem = [
    ("بازار گئی تھیں", "had gone (fem)"),
    ("رات تک کام کرتی رہیں", "kept working (fem)"),
]

for action, _ in hum_actions_fem:
    sentence = f"ہم {action}"
    templates.append((sentence, "fem_plur", "daily"))

# === وہ (he/she/they/formal) patterns ===
wo_pairs = [
    ("وہ اسکول دیر سے پہنچتا ہے", "masc_sing"),
    ("وہ اسکول دیر سے پہنچتی ہے", "fem_sing"),
    ("وہ دفتر جا رہا ہے", "masc_sing"),
    ("وہ دفتر جا رہی ہے", "fem_sing"),
    ("وہ دوست سے ملنے آیا تھا", "masc_sing"),
    ("وہ دوست سے ملنے آئی تھی", "fem_sing"),
    ("وہ موبائل پر گانے سن رہا تھا", "masc_sing"),
    ("وہ موبائل پر گانے سن رہی تھی", "fem_sing"),
    ("وہ کھانا کھا رہا تھا", "masc_sing"),
    ("وہ کھانا کھا رہی تھی", "fem_sing"),
    ("وہ کتاب پڑھ رہا ہے", "masc_sing"),
    ("وہ کتاب پڑھ رہی ہے", "fem_sing"),
    ("وہ بازار گیا تھا", "masc_sing"),
    ("وہ بازار گئی تھی", "fem_sing"),
    ("وہ بازار گئے تھے", "masc_plur"),
    ("وہ بازار گئی تھیں", "fem_plur"),
    ("وہ روزانہ آتا ہے", "masc_sing"),
    ("وہ روزانہ آتی ہے", "fem_sing"),
    ("وہ روزانہ آتے ہیں", "masc_plur"),
    ("وہ روزانہ آتی ہیں", "fem_plur"),
]

for sentence, gender in wo_pairs:
    templates.append((sentence, gender, "daily"))

# === Ergative patterns (نے) ===
ergative_pairs = [
    ("پولیس نے رپورٹ درج کی", "fem_sing"),
    ("حکومت نے اعلان کیا", "masc_sing"),
    ("حکومت نے سہولتیں شروع کی ہیں", "fem_sing"),
    ("لڑکے نے کتاب پڑھی", "fem_sing"),
    ("لڑکے نے اخبار پڑھا", "masc_sing"),
    ("لڑکی نے خط لکھا", "masc_sing"),
    ("لڑکی نے چٹھی لکھی", "fem_sing"),
    ("استاد نے سبق پڑھایا", "masc_sing"),
    ("ماں نے کھانا پکایا", "masc_sing"),
    ("باپ نے بیٹی کو سمجھایا", "masc_sing"),
    ("وزیر نے پالیسی کا اعلان کیا", "masc_sing"),
    ("وزیر نے رپورٹ پیش کی", "fem_sing"),
    ("ٹیم نے میچ جیت لیا", "masc_sing"),
    ("سرکار نے نوکریاں دی ہیں", "fem_sing"),
    ("کمپنی نے مصنوعات فروخت کیں", "fem_sing"),
    ("فوج نے علاقہ خالی کرا لیا", "masc_sing"),
    ("محکمے نے جانچ شروع کی", "fem_sing"),
    ("استانی نے طالبہ کو پڑھایا", "masc_sing"),
    ("بھائی نے بہن کو بلایا", "masc_sing"),
    ("لڑکیوں نے گانا گایا", "masc_sing"),
    ("بچوں نے شور مچایا", "masc_sing"),
    ("ڈاکٹر نے مریض کو دیکھا", "masc_sing"),
    ("نرس نے مریض کو دوا دی", "fem_sing"),
    ("جج نے فیصلہ سنا دیا", "masc_sing"),
    ("افسر نے حکم جاری کیا", "masc_sing"),
]

for sentence, gender in ergative_pairs:
    templates.append((sentence, gender, "news"))

# === میں (I) patterns ===
main_pairs = [
    ("میں نے ناشتہ نہیں کیا ہے", "masc_sing"),
    ("میں نے کھانا کھایا ہے", "masc_sing"),
    ("میں نے نوکری شروع کی ہے", "fem_sing"),
    ("میں نے کتاب خریدی ہے", "fem_sing"),
    ("میں نے خط لکھا ہے", "masc_sing"),
    ("میں نے چٹھی لکھی ہے", "fem_sing"),
    ("میں نے کام شروع کیا ہے", "masc_sing"),
    ("میں نے اسائنمنٹ جمع نہیں کروائی", "fem_sing"),
    ("میں نے رپورٹ جمع کروائی", "fem_sing"),
    ("میں نے درخواست جمع کروائی", "fem_sing"),
    ("میں نے امتحان دیا ہے", "masc_sing"),
    ("میں نے فون کیا تھا", "masc_sing"),
    ("میں نے خط بھیجا تھا", "masc_sing"),
    ("میں بازار جا رہا ہوں", "masc_sing"),
    ("میں بازار جا رہی ہوں", "fem_sing"),
    ("میں نے سبق یاد کر لیا ہے", "masc_sing"),
    ("میں نے چائے پی لی ہے", "fem_sing"),
    ("میں نے فلم دیکھی ہے", "fem_sing"),
    ("میں نے کتاب پڑھ لی ہے", "fem_sing"),
]

for sentence, gender in main_pairs:
    templates.append((sentence, gender, "daily"))

# === Natural phenomena (mostly feminine) ===
nature_pairs = [
    ("بارش سے سڑک بند ہو گئی", "fem_sing"),
    ("بارش ہو رہی ہے", "fem_sing"),
    ("مری میں برف باری ہو رہی ہے", "fem_sing"),
    ("ہوا چل رہی ہے", "fem_sing"),
    ("بجلی چلی گئی", "fem_sing"),
    ("رات ہو گئی ہے", "fem_sing"),
    ("دھوپ نکلی ہے", "fem_sing"),
    ("آندھی آئی تھی", "fem_sing"),
    ("سردی پڑ رہی ہے", "fem_sing"),
    ("گرمی بڑھ گئی ہے", "fem_sing"),
    ("برف پگھل گئی", "fem_sing"),
    ("ژالہ باری ہوئی", "fem_sing"),
    ("کہر چھایا ہوا ہے", "masc_sing"),
    ("طوفان آیا تھا", "masc_sing"),
    ("سیلاب آیا ہے", "masc_sing"),
]

for sentence, gender in nature_pairs:
    templates.append((sentence, gender, "daily"))

# === Adjective-noun agreement ===
adj_noun_pairs = [
    ("یہ کھانا بہت اچھا ہے", "masc_sing"),
    ("یہ کتاب بہت اچھی ہے", "fem_sing"),
    ("یہ تصویریں خوبصورت ہیں", "fem_plur"),
    ("یہ مکان خوبصورت ہے", "masc_sing"),
    ("یہ پھول خوبصورت ہیں", "masc_plur"),
    ("یہ کمرہ بڑا ہے", "masc_sing"),
    ("یہ عمارت بڑی ہے", "fem_sing"),
    ("یہ بچہ ذہین ہے", "masc_sing"),
    ("یہ بچی ذہین ہے", "fem_sing"),
    ("وہ آدمی لمبا ہے", "masc_sing"),
    ("وہ عورت لمبی ہے", "fem_sing"),
    ("یہ پانی ٹھنڈا ہے", "masc_sing"),
    ("یہ چائے ٹھنڈی ہے", "fem_sing"),
    ("وہ موسم گرم ہے", "masc_sing"),
    ("یہ خبر اچھی ہے", "fem_sing"),
    ("یہ کام مشکل ہے", "masc_sing"),
    ("یہ راستہ لمبا ہے", "masc_sing"),
    ("یہ سڑک لمبی ہے", "fem_sing"),
    ("یہ گھر صاف ہے", "masc_sing"),
    ("یہ میز صاف ہے", "fem_sing"),
    ("وہ پھل میٹھا ہے", "masc_sing"),
    ("وہ آم میٹھا ہے", "masc_sing"),
    ("وہ خربوزہ میٹھا ہے", "masc_sing"),
    ("وہ آواز اچھی ہے", "fem_sing"),
    ("یہ تحریر اچھی ہے", "fem_sing"),
    ("وہ فلم اچھی ہے", "fem_sing"),
    ("یہ شہر بڑا ہے", "masc_sing"),
    ("یہ دنیا بڑی ہے", "fem_sing"),
    ("یہ مسئلہ بڑا ہے", "masc_sing"),
    ("یہ مشین تیز ہے", "fem_sing"),
]

for sentence, gender in adj_noun_pairs:
    templates.append((sentence, gender, "general"))

# === Additional verb patterns ===
verb_patterns = [
    # Present tense
    ("لڑکا اسکول جاتا ہے", "masc_sing"),
    ("لڑکی اسکول جاتی ہے", "fem_sing"),
    ("بچے اسکول جاتے ہیں", "masc_plur"),
    ("لڑکیاں اسکول جاتی ہیں", "fem_plur"),
    ("ڈاکٹر کلینک جاتا ہے", "masc_sing"),
    ("استانی اسکول جاتی ہے", "fem_sing"),
    ("کسان کھیت جاتا ہے", "masc_sing"),
    ("طلبہ کالج جاتے ہیں", "masc_plur"),
    # Perfect tense
    ("لڑکا اسکول گیا ہے", "masc_sing"),
    ("لڑکی اسکول گئی ہے", "fem_sing"),
    ("بچے اسکول گئے ہیں", "masc_plur"),
    ("کام ہو گیا ہے", "masc_sing"),
    ("بات ہو گئی ہے", "fem_sing"),
    ("کام ہو گئے ہیں", "masc_plur"),
    # Habitual
    ("میں روزانہ پڑھتا ہوں", "masc_sing"),
    ("میں روزانہ پڑھتی ہوں", "fem_sing"),
    ("وہ روزانہ ورزش کرتا ہے", "masc_sing"),
    ("وہ روزانہ ورزش کرتی ہے", "fem_sing"),
]

for sentence, gender in verb_patterns:
    templates.append((sentence, gender, "daily"))

# === آپ (you formal) ===
aap_pairs = [
    ("آپ نے بہت اچھا کام کیا ہے", "masc_sing"),
    ("آپ نے بہت اچھی بات کی ہے", "fem_sing"),
    ("آپ کب آئے تھے", "masc_plur"),
    ("آپ کب آئی تھیں", "fem_plur"),
    ("آپ کہاں جا رہے ہیں", "masc_plur"),
    ("آپ کہاں جا رہی ہیں", "fem_plur"),
    ("آپ نے کھانا کھایا ہے", "masc_sing"),
]

for sentence, gender in aap_pairs:
    templates.append((sentence, gender, "formal"))

print(f"Total templates: {len(templates)}")

# ─── Generate corruptions ───
# For each template, generate corruptions in all wrong gender/number forms
all_genders = ["masc_sing", "fem_sing", "masc_plur", "fem_plur"]

records = []

for sentence, correct_gender, category in templates:
    # Generate each wrong variant
    corrupted_sentences = set()
    correct_idx = GENDER_ORDER[correct_gender]
    
    for wrong_idx, wrong_gender in enumerate(all_genders):
        if wrong_idx == correct_idx:
            continue
        corrupted = corrupt(sentence, wrong_gender)
        if corrupted and corrupted not in corrupted_sentences:
            corrupted_sentences.add(corrupted)
            records.append({
                "input": f"correct: {corrupted}",
                "output": sentence,
                "category": category,
                "error_type": "grammar",
                "corruption_type": f"{correct_gender}_to_{wrong_gender}",
            })
    
    # Also include the correct sentence as identity (model should not change it)
    if random.random() < 0.12:  # 12% identity rate
        records.append({
            "input": f"correct: {sentence}",
            "output": sentence,
            "category": category,
            "error_type": "grammar",
            "corruption_type": "identity",
        })

random.shuffle(records)
print(f"Total grammar records generated: {len(records)}")

with open("grammar_focus_data.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Saved grammar_focus_data.json")

# Quick check
corr_types = {}
for r in records:
    ct = r["corruption_type"]
    corr_types[ct] = corr_types.get(ct, 0) + 1
print("\nCorruption types distribution:")
for ct, c in sorted(corr_types.items(), key=lambda x: -x[1]):
    print(f"  {ct}: {c}")
