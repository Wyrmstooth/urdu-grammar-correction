# -*- coding: utf-8 -*-
"""
RAG Retriever - Urdu Grammar Rule Matching
Matches actual Urdu text to grammar rule examples using character overlap + keyword matching
"""
import json
import numpy as np
import os

class UrduGrammarRetriever:
    def __init__(self):
        self.rules = self._load_rules()
        self.rule_dict = {r["id"]: r for r in self.rules}
        self._build_index()

    def _load_rules(self):
        with open("urdu_grammar_rules.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self):
        self.examples = [(r["id"], r["example_wrong"]) for r in self.rules]

        # Multilingual model for semantic matching
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            texts = [ex[1] for ex in self.examples]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        except:
            pass

        # Character n-gram TF-IDF (best for Urdu script matching)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=4000,
            lowercase=False,
        )
        texts = [ex[1] for ex in self.examples]
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

        # Error type keywords for direct matching
        self.keyword_map = {
            "spelling": ["muj", "maj", "brs", "bars", "mistake", "typo", "incorrect spelling", "galt", "galti"],
            "extra_words": ["aur aur", "duplicate", "repeat", "extra", "faltu", "redundant", "wah wah", "to to"],
            "missing_words": ["missing", "ko", "mein", "se", "ne", "case marker", "preposition", "nahi", "gayab"],
            "word_order": ["word order", "arrangement", "tarteeb", "pehle", "baad", "structure", "sentence order", "gaya tha", "ke liye"],
            "grammar": ["hain", "hai", "ho", "thi", "tha", "the", "agreement", "gender", "masculine", "feminine", "plural", "singular", "verb", "larki", "bache"],
        }

        print(f"Retriever ready: {len(self.rules)} rules, {len(self.examples)} examples")

    def _char_overlap(self, query):
        """Character n-gram Jaccard overlap between query and examples"""
        def ngrams(s, n=3):
            return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))

        q_ngrams = ngrams(query) | ngrams(query, 4)
        scores = []
        for _, ex in self.examples:
            e_ngrams = ngrams(ex) | ngrams(ex, 4)
            if not e_ngrams or not q_ngrams:
                scores.append(0.0)
                continue
            overlap = len(q_ngrams & e_ngrams)
            scores.append(overlap / max(1, min(len(q_ngrams), len(e_ngrams))))
        arr = np.array(scores)
        return arr / arr.max() if arr.max() > 0 else arr

    def _tfidf_scores(self, query):
        """Char n-gram TF-IDF similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.tfidf.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        sims = np.maximum(sims, 0)
        return sims / sims.max() if sims.max() > 0 else sims

    def _embedding_scores(self, query):
        """Multilingual embedding similarity (if model loaded)"""
        if self.model is None:
            return np.zeros(len(self.examples))
        from sklearn.metrics.pairwise import cosine_similarity
        q_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings).flatten()
        sims = np.maximum(sims, 0)
        return sims / sims.max() if sims.max() > 0 else sims

    def _keyword_boost(self, query, rule_id):
        """Boost score for rules whose keywords appear in query"""
        rule = self.rule_dict[rule_id]
        query_lower = query.lower()
        boost = 1.0
        category = rule.get("category", "")
        if category in self.keyword_map:
            for kw in self.keyword_map[category]:
                if kw in query_lower:
                    boost += 0.15
        return min(boost, 2.0)

    def retrieve(self, query, top_k=3):
        query = query.strip()

        # Three scoring methods
        char_scores = self._char_overlap(query)
        tfidf_scores = self._tfidf_scores(query)
        emb_scores = self._embedding_scores(query)

        # Weighted combination
        combined = 0.5 * char_scores + 0.3 * tfidf_scores + 0.2 * emb_scores

        # Get unique top-k by rule ID
        seen = set()
        scored_rules = []
        for idx in combined.argsort()[::-1]:
            rule_id = self.examples[idx][0]
            if rule_id in seen:
                continue
            seen.add(rule_id)
            base_score = combined[idx]
            boost = self._keyword_boost(query, rule_id)
            scored_rules.append((base_score * boost, rule_id))
            if len(scored_rules) >= top_k * 2:
                break

        scored_rules.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, rule_id in scored_rules[:top_k]:
            results.append({
                "score": float(score),
                "rule": self.rule_dict[rule_id],
            })

        return results

    def build_context(self, query, top_k=3):
        results = self.retrieve(query, top_k=top_k)
        context = ""
        for i, result in enumerate(results, 1):
            rule = result["rule"]
            context += f"[{i}] {rule['rule_en']}\n"
            context += f"    Category: {rule['category']} | Error: {rule['error_type']}\n"
            context += f"    Wrong: {rule['example_wrong']}\n"
            context += f"    Right: {rule['example_right']}\n"
            context += f"    Note: {rule['explanation']}\n\n"
        return context, results


def test():
    r = UrduGrammarRetriever()
    # Test with Romanized for display
    tests = [
        "wo larki bohat acha gati hain",
        "as ne muje fone kiya",
        "bache bahar khel rahe hai",
        "wo ghar gaya aur aur phir wapas aaya",
        "maine bazar sabzi kharidi",
    ]
    for t in tests:
        results = r.retrieve(t, top_k=3)
        cats = [(res['rule']['category'], res['rule']['error_type']) for res in results]
        print(f"\n{t}")
        for i, (c, e) in enumerate(cats, 1):
            print(f"  {i}. [{c}] {e}")


if __name__ == "__main__":
    test()
