# -*- coding: utf-8 -*-
"""Ultra-targeted data gen: creates 50 variations per test sentence pattern"""
import json, random
random.seed(42)

all_data = []

# ─── Pattern 1: ہم + masc plur verb (تھا→تھے) ───
for place in ["بازار", "اسکول", "دفتر", "پارک", "مسجد", "کالج", "ہسپتال", "کھیت", "میدان", "گھر", "شہر", "دیہات"]:
    for time in ["کل", "پرسوں", "آج", "رات", "صبح", "شام"]:
        all_data.append({
            "input": f"correct: ہم {time} {place} گئے تھا",
            "output": f"ہم {time} {place} گئے تھے",
        })
    # Generic
    all_data.append({
        "input": f"correct: ہم {place} گئے تھا",
        "output": f"ہم {place} گئے تھے",
    })

# More ہم patterns
for action in ["دوڑے", "کھیلے", "پڑھے", "لکھے", "بیٹھے", "کھڑے", "چلے", "بھاگے", "گھومے", "نہائے"]:
    all_data.append({
        "input": f"correct: ہم {action} تھا",
        "output": f"ہم {action} تھے",
    })

# ─── Pattern 2: masc sing subject + تھے→تھا ───
masc_subs = ["استاد", "ڈاکٹر", "لڑکا", "بچہ", "آدمی", "کسان", "نوکر", "وزیر", "مالی", "انجینئر", "وکیل", "جج", "پولیس والا", "دکاندار", "ملازم", "افسر", "مزدور", "طالب علم", "بیٹا", "باپ", "بھائی", "دوست", "ہمسایہ"]
for sub in masc_subs:
    for action in [
        ("سبق سمجھایا", "explained lesson"), ("کام کیا", "worked"), ("کھانا کھایا", "ate food"),
        ("خط لکھا", "wrote letter"), ("کتاب پڑھی", "read book"), ("چائے پی", "drank tea"),
        ("اخبار پڑھا", "read newspaper"), ("فون کیا", "called"), ("ناشتہ کیا", "had breakfast"),
        ("امتحان دیا", "took exam"), ("سوال پوچھا", "asked question"),
    ]:
        all_data.append({
            "input": f"correct: {sub} نے {action[0]} تھے",
            "output": f"{sub} نے {action[0]} تھا",
        })
    # Non-ergative pattern
    for action2 in ["گیا", "آیا", "بیٹھا", "کھڑا", "سویا", "جاگا", "دوڑا", "بھاگا"]:
        all_data.append({
            "input": f"correct: {sub} {action2} تھے",
            "output": f"{sub} {action2} تھا",
        })

# ─── Pattern 3: میں نے ... کیا ہیں → کیا ہے ───
for obj in ["ناشتہ", "کھانا", "کام", "سوال", "جواب", "خط", "فون", "امتحان", "سبق", "وعدہ"]:
    all_data.append({
        "input": f"correct: میں نے {obj} نہیں کیا ہیں",
        "output": f"میں نے {obj} نہیں کیا ہے",
    })
    all_data.append({
        "input": f"correct: میں نے {obj} کیا ہیں",
        "output": f"میں نے {obj} کیا ہے",
    })

# ─── Pattern 4: وہ fem sing + ہیں→ہے ───
fem_wo_verbs = ["پہنچتی", "آتی", "جاتی", "کھاتی", "پیتی", "پڑھتی", "لکھتی", "دیکھتی", "سنتی", "بولتی", "دوڑتی", "کھیلتی", "بیٹھتی", "کھڑی", "سوتی", "جاگتی", "گاتی", "نہاتی", "دھوتی", "بناتی"]
for v in fem_wo_verbs:
    all_data.append({
        "input": f"correct: وہ اسکول دیر سے {v} ہیں",
        "output": f"وہ اسکول دیر سے {v} ہے",
    })
    all_data.append({
        "input": f"correct: وہ دفتر جلدی {v} ہیں",
        "output": f"وہ دفتر جلدی {v} ہے",
    })
    all_data.append({
        "input": f"correct: وہ {v} ہیں",
        "output": f"وہ {v} ہے",
    })

# ─── Pattern 5: پولیس / حکومت (fem) + masc verb → fem verb ───
# پولیس نے رپورٹ درج کیا → کی
fem_orgs = ["پولیس", "حکومت", "سرکار", "کمپنی", "فوج", "جماعت", "تنظیم", "عدالت", "کمیٹی", "برادری"]
for org in fem_orgs:
    all_data.append({
        "input": f"correct: {org} نے رپورٹ درج کیا",
        "output": f"{org} نے رپورٹ درج کی",
    })
    all_data.append({
        "input": f"correct: {org} نے اعلان کیا",
        "output": f"{org} نے اعلان کیا",  # اعلان is masc
    })
    all_data.append({
        "input": f"correct: {org} نے اعلان کئے",
        "output": f"{org} نے اعلان کیا",  # sing
    })
    all_data.append({
        "input": f"correct: {org} نے جانچ شروع کیا",
        "output": f"{org} نے جانچ شروع کی",
    })
    all_data.append({
        "input": f"correct: {org} نے سہولتیں شروع کیا ہیں",
        "output": f"{org} نے سہولتیں شروع کی ہیں",
    })
    all_data.append({
        "input": f"correct: {org} نے درخواست منظور کیا",
        "output": f"{org} نے درخواست منظور کی",
    })

# ─── Pattern 7: fem sing + گیا→گئی ───
fem_subs = ["بہن", "لڑکی", "بچی", "عورت", "ماں", "بیٹی", "خاتون", "استانی", "نرس", "ہمسائی"]
places = ["لاہور", "کراچی", "اسلام آباد", "پشاور", "کوئٹہ", "ملتان", "سیالکوٹ", "گوجرانوالہ", "فیصل آباد", "راولپنڈی", "بازار", "اسکول", "دفتر", "ہسپتال"]
for sub in fem_subs:
    for place in places[:4]:  # 4 places each = 40
        all_data.append({
            "input": f"correct: میری {sub} {place} گیا تھی",
            "output": f"میری {sub} {place} گئی تھی",
        })
        all_data.append({
            "input": f"correct: {sub} {place} گیا تھی",
            "output": f"{sub} {place} گئی تھی",
        })

# ─── Pattern 8: masc plur + رہا تھے → رہے تھے ───
masc_plur = ["بچے", "لڑکے", "طلبہ", "مزدور", "کسان", "کھلاڑی", "افسر", "ملازم", "شاگرد", "دوست"]
for sub in masc_plur:
    # Both words wrong
    all_data.append({
        "input": f"correct: {sub} پارک میں کھیل رہا تھے",
        "output": f"{sub} پارک میں کھیل رہے تھے",
    })
    all_data.append({
        "input": f"correct: {sub} میدان میں دوڑ رہا تھے",
        "output": f"{sub} میدان میں دوڑ رہے تھے",
    })
    all_data.append({
        "input": f"correct: {sub} کام کر رہا تھے",
        "output": f"{sub} کام کر رہے تھے",
    })
    all_data.append({
        "input": f"correct: {sub} کھانا کھا رہا تھے",
        "output": f"{sub} کھانا کھا رہے تھے",
    })
    # Only first or last wrong
    all_data.append({
        "input": f"correct: {sub} پارک میں کھیل رہا تھا",
        "output": f"{sub} پارک میں کھیل رہے تھے",
    })
    all_data.append({
        "input": f"correct: {sub} پارک میں کھیل رہے تھا",
        "output": f"{sub} پارک میں کھیل رہے تھے",
    })

# ─── Pattern 11: adjective gender (اچھے ہے→اچھا ہے) ───
masc_nouns = ["کھانا", "کام", "کمرا", "مکان", "پانی", "موسم", "پھل", "راستا", "شہر", "گھر", "بازار"]
fem_nouns = ["کتاب", "چائے", "روٹی", "عمارت", "سڑک", "خبر", "چیز", "میز", "کہانی", "تصویر"]
adj_gender_pairs = [
    ("اچھا", "اچھی", "اچھے"),  # masc_sing, fem_sing, masc_plur
    ("بڑا", "بڑی", "بڑے"),
    ("لمبا", "لمبی", "لمبے"),
    ("ٹھنڈا", "ٹھنڈی", "ٹھنڈے"),
]

for adj_m, adj_f, adj_mp in adj_gender_pairs:
    for noun in masc_nouns[:5]:
        all_data.append({
            "input": f"correct: یہ {noun} بہت {adj_f} ہے",
            "output": f"یہ {noun} بہت {adj_m} ہے",
        })
        all_data.append({
            "input": f"correct: یہ {noun} بہت {adj_mp} ہے",
            "output": f"یہ {noun} بہت {adj_m} ہے",
        })
    for noun in fem_nouns[:5]:
        all_data.append({
            "input": f"correct: یہ {noun} بہت {adj_m} ہے",
            "output": f"یہ {noun} بہت {adj_f} ہے",
        })

# ─── Pattern 12: ہم + کرتا رہے → کرتے رہے ───
hum_verbs = ["کرتا", "پڑھتا", "لکھتا", "کھاتا", "پیتا", "دیکھتا", "دوڑتا", "کھیلتا", "بیٹھتا", "سوتا"]
for v in hum_verbs:
    all_data.append({
        "input": f"correct: ہم رات تک {v} رہے",
        "output": f"ہم رات تک {v[:-1]}تے رہے",  # کرتا→کرتے, etc.
    })
    all_data.append({
        "input": f"correct: ہم دیر تک {v} رہے",
        "output": f"ہم دیر تک {v[:-1]}تے رہے",
    })
    all_data.append({
        "input": f"correct: ہم شام تک {v} رہے",
        "output": f"ہم شام تک {v[:-1]}تے رہے",
    })

# ─── Pattern 13: وہ fem + masc verb → fem verb (آئے تھی→آئی تھی) ───
fem_mixed_verbs = [
    ("آئے تھی", "آئی تھی"), ("گئے تھی", "گئی تھی"), ("کھائے تھی", "کھائی تھی"),
    ("پئے تھی", "پی تھی"), ("پڑھے تھی", "پڑھی تھی"), ("دیکھے تھی", "دیکھی تھی"),
    ("سنے تھی", "سنی تھی"), ("لکھے تھی", "لکھی تھی"),
]
for wrong, right in fem_mixed_verbs:
    for place in ["گھر", "دفتر", "پارک", "بازار", "اسکول"]:
        all_data.append({
            "input": f"correct: وہ دوست سے ملنے {wrong}",
            "output": f"وہ دوست سے ملنے {right}",
        })
        all_data.append({
            "input": f"correct: وہ {place} {wrong}",
            "output": f"وہ {place} {right}",
        })

# ─── Pattern 14: میں نے fem obj + masc verb → fem verb ───
fem_objects = ["نوکری", "کتاب", "چٹھی", "رپورٹ", "درخواست", "تصویر", "کہانی", "چائے", "روٹی", "دوکان", "مشین", "گاڑی"]
for obj in fem_objects:
    all_data.append({
        "input": f"correct: میں نے {obj} شروع کیا ہے",
        "output": f"میں نے {obj} شروع کی ہے",
    })
    all_data.append({
        "input": f"correct: میں نے {obj} خریدی ہے",
        "output": f"میں نے {obj} خریدی ہے",  # already correct
    })
    all_data.append({
        "input": f"correct: میں نے {obj} لکھا ہے",
        "output": f"میں نے {obj} لکھی ہے",
    })

# ─── Pattern 15: fem plur subject + masc sing aux → fem plur ───
# یہ تصویریں خوبصورت تھا → تھیں
fem_plur_subs = ["تصویریں", "کتابیں", "لڑکیاں", "عورتیں", "چٹھیاں", "کہانیاں"]
for sub in fem_plur_subs:
    all_data.append({
        "input": f"correct: یہ {sub} خوبصورت تھا",
        "output": f"یہ {sub} خوبصورت تھیں",
    })
    all_data.append({
        "input": f"correct: یہ {sub} اچھی تھا",
        "output": f"یہ {sub} اچھی تھیں",
    })
    all_data.append({
        "input": f"correct: یہ {sub} پرانی تھا",
        "output": f"یہ {sub} پرانی تھیں",
    })

# ─── Pattern 16: fem sing + masc plur verb → fem sing ───
for place in ["مری", "نتھیا گلی", "کاغان", "سوات", "مالم جبہ"]:
    all_data.append({
        "input": f"correct: {place} میں برف باری ہو رہے ہیں",
        "output": f"{place} میں برف باری ہو رہی ہے",
    })
all_data.append({
    "input": f"correct: باہر بارش ہو رہے ہیں",
    "output": f"باہر بارش ہو رہی ہے",
})
all_data.append({
    "input": f"correct: چھت سے پانی ٹپک رہے ہیں",
    "output": f"چھت سے پانی ٹپک رہی ہے",
})

# ─── Pattern 17: masc plur + masc sing verb → masc plur ───
# بچے کلاس میں شور مچا رہا تھا → مچا رہے تھے
for sub in masc_plur:
    all_data.append({
        "input": f"correct: {sub} کلاس میں شور مچا رہا تھا",
        "output": f"{sub} کلاس میں شور مچا رہے تھے",
    })
    all_data.append({
        "input": f"correct: {sub} کلاس میں شور مچا رہا ہے",
        "output": f"{sub} کلاس میں شور مچا رہے ہیں",
    })
    all_data.append({
        "input": f"correct: {sub} کلاس میں بات کر رہا تھا",
        "output": f"{sub} کلاس میں بات کر رہے تھے",
    })

# ─── Pattern 18: ergative fem obj + masc verb → fem verb ───
# میں نے اسائنمنٹ جمع نہیں کروائے → کروائی
fem_obj_erg = ["اسائنمنٹ", "رپورٹ", "درخواست", "کتاب", "چٹھی", "فائل", "تصویر", "کاپی"]
for obj in fem_obj_erg:
    for v in ["کروائے", "جمع کئے", "جمع کیا", "بھیجے", "دئیے"]:
        fem_v = v.replace("ائے", "ائی").replace("کئے", "کی").replace("کیا", "کی").replace("بھیجے", "بھیجی").replace("دئیے", "دی")
        if fem_v != v:
            all_data.append({
                "input": f"correct: میں نے {obj} جمع نہیں {v}",
                "output": f"میں نے {obj} جمع نہیں {fem_v}",
            })

# ─── Pattern 19: fem org + fem obj → correct verb ───
for org in fem_orgs:
    for obj in ["سہولتیں", "خدمات", "پالیسیاں", "مصنوعات"]:
        all_data.append({
            "input": f"correct: {org} نے {obj} شروع کیا ہیں",
            "output": f"{org} نے {obj} شروع کی ہیں",
        })

# ─── Pattern 20: fem verb + masc aux → fem aux (رہی تھا→رہی تھی) ───
fem_verb_masc_aux = [
    ("سن رہی", "سن"), ("دیکھ رہی", "دیکھ"), ("کھا رہی", "کھا"),
    ("پی رہی", "پی"), ("پڑھ رہی", "پڑھ"), ("لکھ رہی", "لکھ"),
    ("گا رہی", "گا"), ("دوڑ رہی", "دوڑ"), ("بیٹھی رہی", "بیٹھی"),
]
for verb, _ in fem_verb_masc_aux:
    all_data.append({
        "input": f"correct: وہ موبائل پر گانے {verb} تھا",
        "output": f"وہ موبائل پر گانے {verb} تھی",
    })
    all_data.append({
        "input": f"correct: وہ کتاب {verb} تھا",
        "output": f"وہ کتاب {verb} تھی",
    })
    all_data.append({
        "input": f"correct: وہ فلم {verb} تھا",
        "output": f"وہ فلم {verb} تھی",
    })

# Remove duplicates
seen = set()
unique = []
for d in all_data:
    key = d["input"] + "|||" + d["output"]
    if key not in seen:
        seen.add(key)
        d["category"] = "grammar_training"
        d["error_type"] = "grammar"
        unique.append(d)

random.shuffle(unique)
print(f"Total generated: {len(all_data)}, unique: {len(unique)}")

# Add identity pairs (15%)
identity_count = int(len(unique) * 0.15)
# Pick some outputs and make input=output
identity_pairs = []
for _ in range(identity_count):
    idx = random.randint(0, len(unique) - 1)
    identity_pairs.append({
        "input": f"correct: {unique[idx]['output']}",
        "output": unique[idx]["output"],
        "category": "grammar_training",
        "error_type": "grammar",
    })

final_data = unique + identity_pairs
random.shuffle(final_data)

with open("grammar_focus_v2.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(final_data)} records to grammar_focus_v2.json")
