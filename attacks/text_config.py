"""
Central registry for text adversarial attacks and models.

Mirrors config.py ATTACK_REGISTRY — each entry carries full metadata
(paper, category, threat model) and a param schema that can drive UI auto-generation.
"""

AVAILABLE_TEXT_MODELS = {
    "[SST-2] BERT Sentiment (textattack)":   "textattack/bert-base-uncased-SST-2",
    "[SST-2] DistilBERT Sentiment":          "distilbert-base-uncased-finetuned-sst-2-english",
    "[AG News] BERT Topic (textattack)":     "textattack/bert-base-uncased-ag-news",
    "[MNLI] BERT NLI (textattack)":          "textattack/bert-base-uncased-MNLI",
    "[Yelp] BERT Sentiment (textattack)":    "textattack/bert-base-uncased-yelp-polarity",
}

DEFAULT_TEXT_MODEL = "textattack/bert-base-uncased-SST-2"

# MLM model used for BERT-Attack, BAE, Clare, and as fallback for embedding neighbours
DEFAULT_MLM_MODEL = "bert-base-uncased"

TEXT_ATTACK_REGISTRY = {
    # ── Character-Level ──────────────────────────────────────────────────
    "DeepWordBug": {
        "cat": "Character-Level",
        "threat": "blackbox",
        "paper": "Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers",
        "authors": "Gao et al.",
        "year": 2018,
        "arxiv": "1801.04354",
        "desc": "Four scoring strategies (Combined, THS, TTS, Replace-1, Random) rank word importance; "
                "greedily applies character perturbations (swap, substitute, delete, insert) to top-k "
                "words. Enforces Levenshtein edit distance ε=30. Paper default: Combined scoring.",
        "params": {
            "scoring_method": {"type": "select", "default": "combined",
                               "options": ["combined", "temporal", "tail", "replaceone", "random"]},
            "max_candidates": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "max_perturbations": {"type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
        },
    },
    "TextBugger": {
        "cat": "Hybrid (Character + Word)",
        "threat": "whitebox/blackbox",
        "paper": "TEXTBUGGER: Generating Adversarial Text Against Real-world Applications",
        "authors": "Li et al.",
        "year": 2019,
        "arxiv": "1812.05271",
        "desc": "Dual-mode attack with 5 perturbations: Insert (space), Delete (char), Swap (adjacent), "
                "Sub-C (homoglyph + keyboard typo), Sub-W (GloVe embedding neighbour, k=5). White-box "
                "(gradient-based) or black-box (sentence + word importance) scoring. Semantic similarity ≥ 0.8.",
        "params": {
            "max_perturbations": {"type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
            "mode": {"type": "select", "default": "black-box", 
                     "options": ["black-box", "white-box"]},
            "strategy": {"type": "select", "default": "combined",
                        "options": ["bug", "word", "combined"]},
            "similarity_threshold": {"type": "float", "default": 0.8, 
                                    "min": 0.5, "max": 1.0, "step": 0.05},
            "max_queries": {"type": "int", "default": 5000, "min": 100, "max": 10000, "step": 100},
            "seed": {"type": "int", "default": None, "min": 0, "max": 999999, "step": 1},
        },
    },
    "HotFlip": {
        "cat": "Token-Level (Gradient)",
        "threat": "whitebox",
        "paper": "HotFlip: White-Box Adversarial Examples for Text Classification",
        "authors": "Ebrahimi et al.",
        "year": 2018,
        "arxiv": "1712.06751",
        "desc": "White-box gradient-based token substitution via first-order Taylor approximation. "
                "Beam search (width=10) explores multiple flip paths. Constraints: "
                "RepeatModification, StopwordModification, MaxWordsPerturbed, "
                "WordEmbeddingDistance (cosine ≥ 0.8), PartOfSpeech matching.",
        "params": {
            "max_flips": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "beam_width": {"type": "int", "default": 10, "min": 1, "max": 20, "step": 1},
            "max_perturbed": {"type": "int", "default": 2, "min": 1, "max": 10, "step": 1},
            "similarity_threshold": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
        },
    },
    "Pruthi2019": {
        "cat": "Character-Level",
        "threat": "blackbox",
        "paper": "Combating Adversarial Misspellings with Robust Word Recognition",
        "authors": "Pruthi et al.",
        "year": 2019,
        "arxiv": "1905.11268",
        "desc": "Simulates common typos: swap adjacent characters, delete characters, insert characters, "
                "and substitute characters with adjacent QWERTY keyboard keys. Practical real-world attack.",
        "params": {
            "max_perturbations": {"type": "int", "default": 1, "min": 1, "max": 10, "step": 1},
        },
    },

    # ── Word-Level ───────────────────────────────────────────────────────
    "TextFooler": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Is BERT Really Robust? A Strong Baseline for Natural Language Attack",
        "authors": "Jin et al.",
        "year": 2020,
        "arxiv": "1907.11932",
        "desc": "Word importance ranking (delete-one, two-case formula) → counter-fitted embedding "
                "neighbours (BERT-MLM fallback) → filtered by word-embedding cosine ≥ δ, strict POS "
                "match, and sentence-level semantic similarity (distilUSE) ≥ threshold.",
        "params": {
            "max_candidates": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.84, "min": 0.5, "max": 1.0, "step": 0.01},
            "embedding_cos_threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "max_perturbation_ratio": {"type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05},
        },
    },
    "BERT-Attack": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "BERT-ATTACK: Adversarial Attack Against BERT Using BERT",
        "authors": "Li et al.",
        "year": 2020,
        "arxiv": "2004.09984",
        "desc": "Feeds ORIGINAL (unmasked) text into BERT MLM and reads predictions at all "
                "positions simultaneously — core innovation vs BAE. Word importance via [UNK] "
                "replacement. Sub-word aware: Cartesian product of per-position top-k predictions "
                "ranked by perplexity for multi-subword words. Official filter_words list (~240 words). "
                "No inline semantic similarity (official uses post-hoc USE evaluation).",
        "params": {
            "max_candidates": {"type": "int", "default": 48, "min": 5, "max": 200, "step": 5},
            "max_perturbation_ratio": {"type": "float", "default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05},
            "threshold_pred_score": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
            "use_bpe": {"type": "bool", "default": True},
        },
    },
    "BAE": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "BAE: BERT-based Adversarial Examples for Text Classification",
        "authors": "Garg & Ramakrishnan",
        "year": 2020,
        "arxiv": "2004.01970",
        "desc": "Four strategies using BERT MLM — Replace (R): mask word and fill, "
                "Insert (I): insert [MASK] left or right and fill, R/I: pick best "
                "single operation, R+I: sequential replace then insert. POS filter "
                "(allow verb↔noun swap), windowed USE similarity (window=15, "
                "threshold=0.8). Delete (D) is a SarabCraft extension not in the "
                "original paper.",
        "params": {
            "strategy": {"type": "select", "default": "R", "options": ["R", "I", "R/I", "R+I", "D"]},
            "max_candidates": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
            "max_perturbation_ratio": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
        },
    },
    "PWWS": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency",
        "authors": "Ren et al.",
        "year": 2019,
        "arxiv": "1907.06292",
        "desc": "Probability Weighted Word Saliency: H = ΔP × softmax(S), where "
                "S = P(y|x) − P(y|x\\w). WordNet synonyms filtered by Penn Treebank "
                "POS whitelist with pre-filter (multi-word, lemma, POS match). "
                "Greedy substitution in descending H-score order.",
        "params": {
            "max_candidates": {"type": "int", "default": 10, "min": 1, "max": 50, "step": 1},
        },
    },
    "Alzantot GA": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Generating Natural Language Adversarial Examples",
        "authors": "Alzantot et al.",
        "year": 2018,
        "arxiv": "1804.07998",
        "desc": "Genetic algorithm: population of perturbed texts → crossover + mutation "
                "(word substitution) → fitness selection (target confidence). Evolutionary search.",
        "params": {
            "population_size": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_generations": {"type": "int", "default": 40, "min": 5, "max": 200, "step": 5},
            "mutation_rate": {"type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05},
            "similarity_threshold": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
        },
    },
    "Faster Alzantot GA": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Certified Robustness to Adversarial Word Substitutions",
        "authors": "Jia et al.",
        "year": 2019,
        "arxiv": "1909.00986",
        "desc": "Optimized genetic algorithm 10-20x faster than original Alzantot. Uses counter-fitted "
                "embeddings for word substitution with language model scoring for fluency.",
        "params": {
            "population_size": {"type": "int", "default": 60, "min": 10, "max": 200, "step": 10},
            "max_generations": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_perturbation_ratio": {"type": "float", "default": 0.2, "min": 0.05, "max": 1.0, "step": 0.05},
        },
    },
    "IGA": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Natural Language Adversarial Attacks and Defenses in Word Level",
        "authors": "Wang et al.",
        "year": 2019,
        "arxiv": "1909.06723",
        "desc": "Improved Genetic Algorithm with prioritized word importance ranking and enhanced "
                "search strategy. More efficient than original Alzantot with better convergence.",
        "params": {
            "population_size": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_generations": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_perturbation_ratio": {"type": "float", "default": 0.2, "min": 0.05, "max": 1.0, "step": 0.05},
        },
    },
    "PSO": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Word-level Textual Adversarial Attacking as Combinatorial Optimization",
        "authors": "Zang et al.",
        "year": 2020,
        "arxiv": "2004.14641",
        "desc": "Particle Swarm Optimization with sememe-based word substitution. Treats adversarial "
                "text generation as combinatorial optimization with swarm intelligence search.",
        "params": {
            "num_particles": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_iterations": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_perturbation_ratio": {"type": "float", "default": 0.2, "min": 0.05, "max": 1.0, "step": 0.05},
        },
    },

    # ── Sentence-Level ───────────────────────────────────────────────────
    "Clare": {
        "cat": "Sentence-Level",
        "threat": "blackbox",
        "paper": "CLARE: Contextualized Perturbation for Textual Adversarial Attack",
        "authors": "Li et al.",
        "year": 2021,
        "arxiv": "2009.07502",
        "desc": "Contextual perturbation using BERT MLM for Replace, Insert, and Merge "
                "operations. Searches all candidates, selects one maximising label change + fluency.",
        "params": {
            "max_perturbations": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "similarity_threshold": {"type": "float", "default": 0.7, "min": 0.4, "max": 1.0, "step": 0.05},
        },
    },
    "Back-Translation": {
        "cat": "Sentence-Level",
        "threat": "blackbox",
        "paper": "Semantically Equivalent Adversarial Rules for Debugging NLP Models",
        "authors": "Ribeiro et al.",
        "year": 2018,
        "arxiv": "1804.06508",
        "desc": "Paraphrase via translation round-trip: English → pivot language → English "
                "using MarianMT. Checks if paraphrased text flips classifier label.",
        "params": {
            "num_paraphrases": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "similarity_threshold": {"type": "float", "default": 0.6, "min": 0.3, "max": 1.0, "step": 0.05},
        },
    },
}
