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
        "desc": "Scores word importance via delete-one, then applies character perturbations "
                "(swap adjacent, substitute nearby-key, delete, insert) to top-k important words.",
        "params": {
            "max_candidates": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "max_perturbations": {"type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
        },
    },
    "TextBugger": {
        "cat": "Character-Level",
        "threat": "blackbox",
        "paper": "TEXTBUGGER: Generating Adversarial Text Against Real-world Applications",
        "authors": "Li et al.",
        "year": 2019,
        "arxiv": "1812.05271",
        "desc": "Five character-level perturbations: insert space, delete char, swap adjacent, "
                "substitute homoglyph (visually similar Unicode), substitute nearby keyboard key.",
        "params": {
            "max_perturbations": {"type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
        },
    },
    "HotFlip": {
        "cat": "Character-Level",
        "threat": "whitebox",
        "paper": "HotFlip: White-Box Adversarial Examples for Text Classification",
        "authors": "Ebrahimi et al.",
        "year": 2018,
        "arxiv": "1712.06751",
        "desc": "Gradient-based character flip: computes gradient w.r.t. one-hot character/token "
                "embeddings and finds the substitution that maximises the directional derivative.",
        "params": {
            "max_flips": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "beam_width": {"type": "int", "default": 1, "min": 1, "max": 10, "step": 1},
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
        "desc": "Word importance ranking (delete-one) → counter-fitted embedding neighbours "
                "(BERT-MLM fallback) → filtered by POS match + semantic similarity ≥ threshold.",
        "params": {
            "max_candidates": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
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
        "desc": "Uses BERT masked language model to generate contextually appropriate word "
                "substitutions. Sub-word aware: aggregates WordPiece tokens before replacement.",
        "params": {
            "max_candidates": {"type": "int", "default": 48, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
            "max_perturbation_ratio": {"type": "float", "default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05},
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
                "Insert (I): insert [MASK] adjacent and fill, combined R+I, and Delete (D).",
        "params": {
            "strategy": {"type": "select", "default": "R", "options": ["R", "I", "R+I", "D"]},
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
        "desc": "Probability Weighted Word Saliency: score = ΔP × P(word_saliency). "
                "Uses WordNet synonyms filtered by POS tag, greedy substitution in descending order.",
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
