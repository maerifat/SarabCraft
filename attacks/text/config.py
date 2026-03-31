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

# MLM model used for BERT-Attack, BAE, and as fallback for embedding neighbours
# (Clare uses its own distilroberta-base MLM, matching the original paper)
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
        "desc": "Faithful to the original paper and official QData/deepWordBug code. "
                "Five scoring strategies (Combined, THS, TTS, Replace-1, Random) rank word importance; "
                "applies the chosen character transformer (homoglyph, swap, flip, remove, insert) to "
                "the top ε words. Paper defaults: Combined scoring, Homoglyph transformer.",
        "params": {
            "scoring_method": {"type": "select", "default": "combined",
                               "options": ["combined", "temporal", "tail", "replaceone", "random"]},
            "transformer": {"type": "select", "default": "homoglyph",
                            "options": ["homoglyph", "swap", "flip", "remove", "insert"]},
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
        "desc": "Exact match with TextAttack HotFlipEbrahimi2017 recipe. "
                "WordSwapGradientBased(top_n=1): cross-entropy loss gradient, first-order "
                "Taylor approximation, lookup_table via .weight.data, pad_token_id masking, "
                "global top-1 (position, token) selection across flattened score matrix. "
                "BeamSearch(beam_width=10): goal-function scoring via model re-query. "
                "Constraints: RepeatModification, StopwordModification, "
                "MaxWordsPerturbed(2), WordEmbeddingDistance(min_cos_sim=0.8, "
                "counter-fitted GloVe, per-word), PartOfSpeech. "
                "Extension: targeted attack support via target_label.",
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
        "desc": "Faithful to the original paper and TextAttack Pruthi2019 recipe. "
                "Four character-level typo operations — swap adjacent, delete, insert (random a-z), "
                "QWERTY keyboard substitution (4-directional adjacency) — applied only to internal "
                "characters (first and last preserved, matching the psycholinguistic constraint). "
                "Constraints: MinWordLength(4), StopwordModification, RepeatModification, "
                "MaxWordsPerturbed. GreedySearch over all candidates per step.",
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
        "desc": "Exact match to TextAttack TextFoolerJin2019 recipe. "
                "GreedyWordSwapWIR(delete): two-case word importance formula. "
                "WordSwapEmbedding(max_candidates=50): counter-fitted PARAGRAM-SL999 "
                "(BERT-MLM fallback). Constraints: RepeatModification, "
                "StopwordModification (official 280-word list), "
                "WordEmbeddingDistance(min_cos_sim=0.5), "
                "PartOfSpeech(allow_verb_noun_swap=True), "
                "UniversalSentenceEncoder(threshold=0.840845, metric=angular, "
                "compare_against_original=False, window_size=15, "
                "skip_text_shorter_than_window=True).",
        "params": {
            "max_candidates": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.840845057, "min": 0.5, "max": 1.0, "step": 0.01},
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
                "ranked by perplexity for multi-subword words. Official filter_words list (~300 words), "
                "applied to both source words and candidate substitutions. "
                "No inline semantic similarity (official uses post-hoc USE evaluation).",
        "params": {
            "max_candidates": {"type": "int", "default": 48, "min": 5, "max": 200, "step": 5},
            "max_perturbation_ratio": {"type": "float", "default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05},
            "threshold_pred_score": {"type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5},
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
                "threshold=0.936338023). Delete (D) is a SarabCraft extension not in the "
                "original paper.",
        "params": {
            "strategy": {"type": "select", "default": "R", "options": ["R", "I", "R/I", "R+I", "D"]},
            "max_candidates": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.936338023, "min": 0.5, "max": 1.0, "step": 0.01},
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
        "desc": "Exact match to official JHL-HUST/PWWS. Saliency S = P(y|x) − P(y|x\\w) "
                "computed for ALL words; softmax over the FULL saliency vector; "
                "H = ΔP × softmax(S). WordNet synonyms filtered by Penn Treebank POS "
                "whitelist with pre-filter (multi-word, lemma, POS match, 'be'). "
                "Position-stable greedy substitution in descending H-score order. "
                "Optional Named Entity substitution path (dataset-specific NE lists).",
        "params": {
            "max_candidates": {"type": "int", "default": 50, "min": 0, "max": 200, "step": 5,
                               "desc": "Max synonyms evaluated per word (0 = no limit, matching official)"},
            "use_named_entities": {"type": "bool", "default": False,
                                   "desc": "Enable Named Entity substitution (requires NE candidate list)"},
        },
    },
    "Alzantot GA": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Generating Natural Language Adversarial Examples",
        "authors": "Alzantot et al.",
        "year": 2018,
        "arxiv": "1804.07998",
        "desc": "Genetic algorithm with counter-fitted embedding neighbours (Paragram), "
                "softmax fitness-proportional parent selection (T=0.3), uniform crossover, "
                "best-improvement mutation. Inline constraints: RepeatModification, "
                "StopwordModification, MaxWordsPerturbed (20%), WordEmbeddingDistance (≥0.5). "
                "Paper defaults: S=60 population, N=20 generations, δ=0.5.",
        "params": {
            "population_size": {"type": "int", "default": 60, "min": 5, "max": 200, "step": 5},
            "max_generations": {"type": "int", "default": 20, "min": 5, "max": 200, "step": 5},
            "mutation_rate": {"type": "float", "default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05},
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
        "desc": "Optimized Alzantot GA with three modifications: (1) substitutions "
                "pre-computed relative to original (RepeatModification), (2) faster "
                "LM constraint (LearningToWrite, W=6, δ=5.0) compared against original, "
                "(3) 40 iterations. Counter-fitted Paragram embeddings (N=8, δ≤0.5), "
                "softmax fitness-proportional selection (T=0.3), uniform crossover, "
                "best-improvement mutation. Paper defaults: S=60, N=40.",
        "params": {
            "population_size": {"type": "int", "default": 60, "min": 10, "max": 200, "step": 10},
            "max_generations": {"type": "int", "default": 40, "min": 5, "max": 200, "step": 5},
            "mutation_rate": {"type": "float", "default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05},
            "similarity_threshold": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
        },
    },
    "IGA": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Natural Language Adversarial Attacks and Defenses in Word Level",
        "authors": "Wang et al.",
        "year": 2019,
        "arxiv": "1909.06723",
        "desc": "Improved Genetic Algorithm with three innovations over Alzantot GA: "
                "(1) population initialized by replacing each word by its optimal synonym, "
                "(2) max_replace_times_per_index (λ=5) allowing multiple substitutions per "
                "position with remaining budget as selection weight, (3) single-point "
                "crossover. Counter-fitted Paragram embeddings (N=50, δ≤0.5), softmax "
                "fitness-proportional selection (T=0.3), best-improvement mutation. "
                "Paper defaults: S=60, M=20, λ=5.",
        "params": {
            "population_size": {"type": "int", "default": 60, "min": 10, "max": 200, "step": 10},
            "max_generations": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_replace_times_per_index": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
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
        "desc": "Compliant with TextAttack PSOZang2020 recipe and ParticleSwarmOptimization "
                "search method. Discrete velocity via equality function (±V_max=3.0), sigmoid turn "
                "probability, two-phase movement (local/global elite), adaptive ω/C1/C2 schedules, "
                "mutation via best-improvement neighbour replacement, query budget enforcement "
                "(mirrors TextAttack _search_over). Constraints: RepeatModification, "
                "StopwordModification. Transformation: HowNet sememe-based substitution "
                "(WordSwapHowNet) with automatic MLM fallback when the synonym bank is unavailable. "
                "Paper defaults: pop_size=60, max_iters=20, ω₁=0.8, ω₂=0.2.",
        "params": {
            "pop_size": {"type": "int", "default": 60, "min": 5, "max": 200, "step": 5},
            "max_iters": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "max_perturbation_ratio": {"type": "float", "default": 0.2, "min": 0.05, "max": 1.0, "step": 0.05},
            "max_queries": {"type": "int", "default": 5000, "min": 100, "max": 10000, "step": 100},
            "seed": {"type": "int", "default": None, "min": 0, "max": 999999, "step": 1},
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
        "desc": "Contextual perturbation using distilroberta-base MLM for Replace, Insert, and "
                "Merge operations. Global greedy search: generates ALL candidates across ALL "
                "positions and ALL operations per step, selects globally best perturbation. "
                "POS-based merge eligibility, USE similarity (window=15, threshold=0.7), "
                "RepeatModification + StopwordModification constraints.",
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
        "desc": "Paraphrase via translation round-trip using MarianMT ROMANCE multilingual models "
                "(opus-mt-en-ROMANCE / opus-mt-ROMANCE-en). Matches TextAttack BackTranslation: "
                "single pivot (default: Spanish) or chained back-translation through N randomly "
                "sampled pivot languages for stronger perturbation. Language tag prefix (>>lang<< ) "
                "for ROMANCE model format. Semantic similarity filtering (distilUSE cosine ≥ threshold).",
        "params": {
            "num_paraphrases": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "similarity_threshold": {"type": "float", "default": 0.6, "min": 0.3, "max": 1.0, "step": 0.05},
            "chained_back_translation": {"type": "int", "default": 0, "min": 0, "max": 10, "step": 1},
            "target_lang": {"type": "select", "default": "es",
                            "options": ["es", "fr", "it", "pt", "ro", "ca", "gl"]},
        },
    },

    # ── Word-Level (additional) ──────────────────────────────────────────
    "A2T": {
        "cat": "Word-Level",
        "threat": "whitebox",
        "paper": "Towards Improving Adversarial Training of NLP Models",
        "authors": "Yoo & Qi",
        "year": 2021,
        "arxiv": "2109.00544",
        "desc": "Exact match with TextAttack A2TYoo2021 recipe. "
                "GreedyWordSwapWIR(wir_method='gradient'): gradient-based word importance. "
                "Default: WordSwapEmbedding(max_candidates=20) + WordEmbeddingDistance(min_cos_sim=0.8). "
                "MLM variant: WordSwapMaskedLM(method='bae', max_candidates=20, bert-base-uncased). "
                "Constraints: RepeatModification, StopwordModification, "
                "PartOfSpeech(allow_verb_noun_swap=False), "
                "MaxModificationRate(max_rate=0.1, min_threshold=4), "
                "SBERT('stsb-distilbert-base', threshold=0.9, metric='cosine'). "
                "On success: returns candidate with highest similarity score.",
        "params": {
            "mlm": {"type": "bool", "default": False,
                    "desc": "Use A2T-MLM variant (WordSwapMaskedLM) instead of embedding-based"},
            "max_candidates": {"type": "int", "default": 20, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.9, "min": 0.5, "max": 1.0, "step": 0.01},
            "max_modification_rate": {"type": "float", "default": 0.1, "min": 0.05, "max": 1.0, "step": 0.05},
            "embedding_cos_threshold": {"type": "float", "default": 0.8, "min": 0.3, "max": 1.0, "step": 0.05},
        },
    },

    # ── Sentence-Level (additional) ──────────────────────────────────────
    "CheckList": {
        "cat": "Sentence-Level",
        "threat": "blackbox",
        "paper": "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList",
        "authors": "Ribeiro et al.",
        "year": 2020,
        "arxiv": "2005.04118",
        "desc": "Behavioural testing framework (ACL 2020 Best Paper). Builds MFT (unit tests with expected "
                "labels), INV (invariance — prediction must not change under meaning-preserving edits), and "
                "DIR (directional — prediction must change predictably) tests from perturbation generators: "
                "negation, contraction, temporal swap, taxonomy swap, number, typo, punctuation, and NER "
                "entity replacement. Reports per-capability failure rates via a TestSuite.",
        "params": {
            "test_types": {"type": "select", "default": "all",
                           "options": ["all", "mft", "inv", "dir"]},
            "perturbation_types": {"type": "select", "default": "all",
                                   "options": ["all", "negation", "contraction", "temporal",
                                               "taxonomy", "number", "typo", "punctuation",
                                               "entity_name", "entity_location"]},
            "max_test_cases": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5},
            "similarity_threshold": {"type": "float", "default": 0.7, "min": 0.3, "max": 1.0, "step": 0.05},
        },
    },
    "StressTest": {
        "cat": "Sentence-Level",
        "threat": "blackbox",
        "paper": "Stress Test Evaluation for Natural Language Inference",
        "authors": "Naik et al.",
        "year": 2018,
        "arxiv": "1806.00692",
        "desc": "Concatenation-based attack appending distraction sentences: tautology "
                "('and true is true'), negation ('and false is not true'), overlap (duplicate "
                "input chunk), length (neutral filler padding), noise (unrelated sentences). "
                "Tests whether models are distracted by irrelevant appended content — models "
                "that rely on keyword matching or length bias are vulnerable. NAACL 2018.",
        "params": {
            "strategies": {"type": "select", "default": "all",
                           "options": ["all", "tautology", "negation", "overlap",
                                       "length", "noise"]},
            "candidates_per_strategy": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "similarity_threshold": {"type": "float", "default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05},
        },
    },
    "SCPN": {
        "cat": "Sentence-Level",
        "threat": "blackbox",
        "paper": "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks",
        "authors": "Iyyer et al.",
        "year": 2018,
        "arxiv": "1804.06516",
        "desc": "Syntactic paraphrase attack: generates paraphrases conforming to different "
                "syntactic structures via Pegasus-paraphrase (modern drop-in for original SCPN model). "
                "Diverse beam search + nucleus sampling produce syntactically varied candidates. "
                "Semantic similarity filtering ensures meaning preservation. Stronger than simple "
                "back-translation because syntax is actively diversified.",
        "params": {
            "num_paraphrases": {"type": "int", "default": 10, "min": 1, "max": 30, "step": 1},
            "similarity_threshold": {"type": "float", "default": 0.7, "min": 0.3, "max": 1.0, "step": 0.05},
            "temperature": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1},
            "top_p": {"type": "float", "default": 0.95, "min": 0.5, "max": 1.0, "step": 0.05},
        },
    },

    # ── Universal (Trigger-Based) ────────────────────────────────────────
    "UAT": {
        "cat": "Universal",
        "threat": "whitebox",
        "paper": "Universal Adversarial Triggers for Attacking and Analyzing NLP",
        "authors": "Wallace et al.",
        "year": 2019,
        "arxiv": "1908.07125",
        "desc": "Gradient-based discrete optimisation that finds a short token sequence (trigger) "
                "causing misclassification when prepended or appended to input. First-order Taylor "
                "approximation scores all vocabulary tokens at each trigger position. Per-input mode "
                "optimises trigger for specific input; universal mode (via batch jobs) aggregates "
                "gradients across inputs. EMNLP 2019.",
        "params": {
            "trigger_length": {"type": "int", "default": 3, "min": 1, "max": 10, "step": 1},
            "num_iterations": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 5},
            "beam_size": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1},
            "position": {"type": "select", "default": "prepend", "options": ["prepend", "append"]},
        },
    },

    # ── Deletion-Based ───────────────────────────────────────────────────
    "Input Reduction": {
        "cat": "Deletion",
        "threat": "blackbox",
        "paper": "Pathologies of Neural Models Make Interpretations Difficult",
        "authors": "Feng et al.",
        "year": 2018,
        "arxiv": "1804.07781",
        "desc": "The only deletion-based attack: iteratively removes the least important word "
                "until prediction changes or minimal sufficient input is reached. Exposes models "
                "that maintain predictions even after most content is removed, revealing "
                "pathological over-confidence and reliance on spurious features. EMNLP 2018.",
        "params": {
            "max_reduction_ratio": {"type": "float", "default": 0.7, "min": 0.1, "max": 0.95, "step": 0.05},
            "stop_at_length": {"type": "int", "default": 1, "min": 1, "max": 10, "step": 1},
        },
    },

    # ── Word-Level (perplexity-constrained) ──────────────────────────────
    "Kuleshov2017": {
        "cat": "Word-Level",
        "threat": "blackbox",
        "paper": "Adversarial Examples for Natural Language Classification Problems",
        "authors": "Kuleshov et al.",
        "year": 2018,
        "arxiv": "1707.05461",
        "desc": "Greedy word substitution constrained by GPT-2 language model perplexity "
                "instead of sentence-embedding similarity (USE). Candidates from counter-fitted "
                "embeddings are accepted only if they don't increase perplexity beyond a ratio "
                "threshold. Produces more grammatically natural adversarial examples than "
                "USE-constrained attacks. TextAttack Kuleshov2017 recipe.",
        "params": {
            "max_candidates": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5},
            "max_perplexity_ratio": {"type": "float", "default": 4.0, "min": 1.5, "max": 10.0, "step": 0.5},
            "max_perturbation_ratio": {"type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05},
            "embedding_cos_threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
        },
    },

    # ── Embedding-Space (Projected Gradient) ─────────────────────────────
    "Seq2Sick": {
        "cat": "Embedding-Space",
        "threat": "whitebox",
        "paper": "Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples",
        "authors": "Cheng et al.",
        "year": 2020,
        "arxiv": "1803.01128",
        "desc": "Projected gradient attack operating in continuous EMBEDDING SPACE — the only "
                "attack that applies PGD directly to token embeddings then projects back to "
                "nearest discrete tokens. Enables gradient-based optimization impossible with "
                "purely discrete methods. Adapted from seq2seq formulation for classification. "
                "ICLR 2020.",
        "params": {
            "num_iterations": {"type": "int", "default": 30, "min": 5, "max": 100, "step": 5},
            "step_size": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.005},
            "max_perturbation_ratio": {"type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05},
            "similarity_threshold": {"type": "float", "default": 0.7, "min": 0.3, "max": 1.0, "step": 0.05},
        },
    },

    # ── Morphological ────────────────────────────────────────────────────
    "MorpheuS": {
        "cat": "Morphological",
        "threat": "blackbox",
        "paper": "It's Morphin' Time! Combating Linguistic Discrimination with Inflectional Perturbations",
        "authors": "Tan et al.",
        "year": 2020,
        "arxiv": "2009.11112",
        "desc": "Inflectional morphology attack: changes grammatical inflection (verb tense, "
                "noun number, adjective degree) rather than substituting synonyms or introducing "
                "typos. Linguistically principled: 'walked'→'walks', 'cats'→'cat', 'better'→'best'. "
                "Uses NLTK lemmatization + rule-based inflection generation. Tests whether models "
                "are sensitive to valid grammatical form variations.",
        "params": {
            "max_perturbation_ratio": {"type": "float", "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05},
            "similarity_threshold": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.05},
        },
    },
}
