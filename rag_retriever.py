# -*- coding: utf-8 -*-
"""
RAG Retriever v2 - Smarter rule matching using word overlap + keyword classification.
No more char-level hallucination.
"""
import json
import re
import numpy as np

class UrduGrammarRetriever:
    def __init__(self):
        self.rules = self._load_rules()
        self.rule_dict = {r["id"]: r for r in self.rules}
        print(f"Retriever ready: {len(self.rules)} rules")

    def _load_rules(self):
        with open("urdu_grammar_rules.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_words(self, text):
        """Extract unique words from Urdu text."""
        return set(re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', text))

    def _word_overlap_score(self, query_words, example_words):
        """Jaccard similarity on words."""
        if not query_words or not example_words:
            return 0.0
        intersection = query_words & example_words
        union = query_words | example_words
        return len(intersection) / len(union) if union else 0.0

    def _keyword_classify(self, text):
        """Classify what error types are likely present based on keywords."""
        text_words = set(re.findall(r'[\u0600-\u06FF\u0750-\u077F]+', text))
        scores = {}
        
        # Spelling indicators: common misspelled patterns
        spelling_patterns = {"مجے", "بارس", "جوگنگ", "گئ", "ہوئ", "لذیذ", "سبز", "کرتہ", "کھیل", "بہت"}
        # Word order indicators: verb before object patterns  
        word_order_patterns = {"لیے", "کے", "سے", "پہلے", "بعد"}
        # Extra words indicators
        extra_patterns = {"اور", "بہت", "خوب", "بہت زیادہ"}
        # Missing words indicators
        missing_patterns = {"میں", "نے", "کو", "سے", "پر", "کا", "کی", "کے"}
        # Grammar indicators: verb endings
        grammar_patterns = {"تھا", "تھی", "تھے", "تھیں", "ہے", "ہیں", "ہوں", "رہا", "رہے", "رہی", "رہیں",
                           "گیا", "گئی", "گئے", "گیا", "کرتا", "کرتے", "کرتی", "جاتا", "جاتے", "جاتی",
                           "کیا", "کی", "کئے", "ہوا", "ہوئی", "ہوئے"}

        for category, patterns in [
            ("spelling", spelling_patterns),
            ("word_order", word_order_patterns),
            ("extra_words", extra_patterns),
            ("missing_words", missing_patterns),
            ("grammar", grammar_patterns),
        ]:
            overlap = text_words & patterns
            if overlap:
                scores[category] = len(overlap)

        # Boost grammar if no other strong signals (most user sentences are grammar)
        if not scores:
            scores["grammar"] = 1

        return scores

    def retrieve(self, query, top_k=4):
        if not query or not query.strip():
            return []

        query = query.strip()
        query_words = self._extract_words(query)
        keyword_scores = self._keyword_classify(query)

        scored = []
        for rule in self.rules:
            rule_wrong = rule.get("example_wrong", "")
            rule_right = rule.get("example_right", "")
            rule_category = rule.get("category", "")
            error_type = rule.get("error_type", "")

            wrong_words = self._extract_words(rule_wrong)
            right_words = self._extract_words(rule_right)
            all_rule_words = wrong_words | right_words

            # Word overlap with both query and rule examples
            word_score = max(
                self._word_overlap_score(query_words, wrong_words),
                self._word_overlap_score(query_words, all_rule_words),
            )

            # Category boost from keyword classification
            cat_boost = keyword_scores.get(rule_category, 0) * 0.2
            gen_boost = keyword_scores.get("grammar", 0) * 0.1  # grammar always gets small boost

            # Penalize identity pairs (wrong == right)
            is_identity = (rule_wrong.strip() == rule_right.strip())

            total_score = word_score * 5.0 + cat_boost * 0.3 + gen_boost * 0.2
            if is_identity:
                total_score -= 10.0
            # Penalize if rule example has fewer than 2 overlapping words with query
            if len(query_words & all_rule_words) < 2:
                total_score -= 3.0

            scored.append((total_score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_ids = set()
        for score, rule in scored:
            if rule["id"] in seen_ids:
                continue
            if score <= 2.0:
                continue
            seen_ids.add(rule["id"])
            results.append({"score": float(score), "rule": rule})
            if len(results) >= top_k:
                break

        # If we got nothing useful, return top grammar rules
        if not results:
            grammar_rules = [r for r in self.rules if r.get("category") == "grammar" and r.get("example_wrong") != r.get("example_right")]
            for r in grammar_rules[:top_k]:
                results.append({"score": 0.5, "rule": r})

        return results

    def build_context(self, query, top_k=4):
        results = self.retrieve(query, top_k=top_k)
        context = ""
        for i, result in enumerate(results, 1):
            rule = result["rule"]
            context += f"[{i}] {rule['rule_en']}\n"
            context += f"    Wrong: {rule['example_wrong']}\n"
            context += f"    Right: {rule['example_right']}\n"
            context += f"    Note: {rule['explanation']}\n\n"
        return context, results
