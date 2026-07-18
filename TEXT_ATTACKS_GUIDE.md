# Text Adversarial Attacks — A Practical Guide

> From zero to running attacks in SarabCraft.  
> Start at Part 1 if you're new. Jump to Part 3 if you just want to run attacks now.

---

## Part 1 — How Transformers Work (the simple version)

### 1.1 What is a text classifier?

You give it a sentence. It gives you a label.

```
Input:  "This movie was absolutely terrible"
Output: NEGATIVE (98% confident)
```

That's it. Under the hood, a transformer model like BERT does three things:

**Step 1 — Tokenize**  
Split the text into tokens (subword pieces the model was trained on).

```
"This movie was absolutely terrible"
→ ["this", "movie", "was", "absolutely", "ter", "##rible"]
   [  101,   3185,   2001,    4952,       8915,    ##1890]
```

Each token gets a number (a token ID) from the model's vocabulary (~30,000 words).

**Step 2 — Embed**  
Each token ID is looked up in a giant table to get a 768-number vector. Think of it as a point in 768-dimensional space. Similar words are near each other.

```
token ID 3185  →  [0.21, -0.04, 0.88, ..., 0.33]  ← "movie"
token ID 8915  →  [-0.71, 0.55, 0.12, ..., -0.44] ← "ter" (part of "terrible")
```

**Step 3 — Attend and classify**  
The transformer runs 12 layers of "attention" — every token looks at every other token and updates its own representation based on context. At the end, a small linear layer maps the final representation to 2 numbers (for a 2-class model):

```
logits = [2.1, -2.1]  →  softmax  →  [0.98, 0.02]
                                        ↑ NEGATIVE    ↑ POSITIVE
```

**The key insight:** the model only ever sees token IDs. It has no idea what the original characters were. This is the attack surface.

---

### 1.2 Why are transformers vulnerable?

The model was trained on clean text. It built up patterns like:

> "If I see the token for *terrible*, the output should be NEGATIVE."

An adversarial attack breaks this by making the model see *different tokens* while you as a human still read the same words.

Three ways to do this:

| Method | What you change | What the model sees |
|--------|----------------|---------------------|
| **Character-level** | `terrible` → `tеrriblе` (Unicode swap) | Completely unknown token sequence |
| **Word-level** | `terrible` → `dreadful` | A different but similar token |
| **Sentence-level** | Paraphrase the whole sentence | Different tokens, same meaning |

---

## Part 2 — The Attack Framework

Every attack in SarabCraft has the same skeleton:

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT TEXT                           │
│       "This movie was absolutely terrible"             │
└────────────────────┬────────────────────────────────────┘
                     │
              ┌──────▼──────┐
              │  SCORE      │  Which words matter most to the model?
              │  WORDS      │  (query model, measure importance)
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  FIND       │  What can we replace them with?
              │  CANDIDATES │  (embeddings / MLM / character ops)
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  FILTER     │  Does the replacement still look natural?
              │  CONSTRAINTS│  (semantic similarity, POS match)
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  CHECK      │  Did the model's prediction change?
              │  LABEL FLIP │  If yes → SUCCESS. If no → try next word.
              └──────┬──────┘
                     │
┌─────────────────────▼───────────────────────────────────┐
│                ADVERSARIAL TEXT                         │
│       "This movie was absolutely tеrriblе"             │
│       Model predicts: POSITIVE (61%)  ← fooled!        │
└─────────────────────────────────────────────────────────┘
```

The **AttackResult** the app returns always contains:

| Field | Meaning |
|-------|---------|
| `original_text` | What you typed |
| `adversarial_text` | What fooled the model |
| `original_label` | What the model said before |
| `adversarial_label` | What the model says after |
| `original_confidence` | e.g. 98% |
| `adversarial_confidence` | e.g. 61% |
| `perturbation_ratio` | Fraction of words changed |
| `semantic_similarity` | 0.0–1.0, how similar the two texts are |
| `success` | Did the label actually flip? |

---

## Part 3 — Practical Attack Examples

> **Model used in all examples:** `[SST-2] BERT Sentiment (textattack)`  
> This model classifies text as POSITIVE or NEGATIVE.

---

### Example 1 — DeepWordBug (Character-Level)

**The concept:** Find the most important words, swap one character to a Unicode lookalike. Invisible to humans, unknown to the tokenizer.

**Settings in app:**
```
Attack:           DeepWordBug
Model:            [SST-2] BERT Sentiment (textattack)
scoring_method:   combined
transformer:      homoglyph
max_perturbations: 5
```

**Input text to use:**
```
This movie was absolutely terrible and I hated every single minute of it
```

**What the attack does, step by step:**

```
Step 1 — Score words (Combined THS+TTS):
  "terrible"   → importance 0.41  ← highest, this word drives the prediction
  "absolutely" → importance 0.12
  "hated"      → importance 0.09
  "single"     → importance 0.04
  "movie"      → importance 0.02

Step 2 — Apply homoglyph transformer to top 3 words:
  "terrible"   → look up each character in HOMOGLYPHS table
                 'e' → 'е' (Cyrillic), pick a random position
                 result: "tеrriblе"
  "absolutely" → 'a' → 'ɑ' (Cyrillic alpha)
                 result: "ɑbsolutely"
  "hated"      → 'a' → 'ɑ'
                 result: "hɑted"

Step 3 — Output (looks identical to you):
  "This movie was ɑbsolutely tеrriblе and I hɑted every single minute of it"

Step 4 — Model now classifies: POSITIVE (61%)  ← fooled
```

**Why homoglyphs work:**  
BERT's tokenizer turns `tеrriblе` into `[UNK]` or some garbage subword sequence — the model has *never seen this token during training*. Without its strongest signal word, it makes a wrong prediction.

**Try varying the transformer:**
- `swap`: `"terrible"` → `"etrriblel"` — transposes two adjacent letters, still often fools the model
- `remove`: `"terrible"` → `"terribl"` — deletes one character
- `flip`: `"terrible"` → `"terribme"` — substitutes one letter with a random other (a–z, excluding original)

**Confirmed live result (reproduced in SarabCraft):**

```
Settings:  scoring=combined  transformer=homoglyph  max_perturbations=3
Input:     Scientists have discovered a new exoplanet with signs of water
```

```
Step 1 — Combined scoring ranks content words first:
  "Scientists" → highest (drives Sci/Tech positive framing)
  "discovered" → second
  "water"      → third
  "have", "a", "with", "of" → near-zero (stopwords, no sentiment signal)

Step 2 — Homoglyph transformer applied to top 3:
  "Scientists" → 't' → '𝚝' (mathematical monospace t)   → "Scientis𝚝s"
  "discovered" → 'o' → 'о' (Cyrillic o)                 → "discоvered"
  "water"      → 'a' → 'ɑ' (IPA alpha)                  → "wɑter"

Step 3 — Adversarial text (looks identical to a human):
  "Scientis𝚝s have discоvered a new exoplanet with signs of wɑter"
```

```
Result:
  Original:    LABEL_1 / POSITIVE  88.3%
  Adversarial: LABEL_0 / NEGATIVE  88.3%  ✓ label flipped

  Perturbation ratio:  30%   (3 of 9 content words changed)
  Semantic similarity: 67%   (homoglyphs confuse the sentence encoder too)
  Queries:             24    (2n for combined, n=9 words — plus initial predict)
  Time:                2.4s
```

> **Note on semantic similarity:** 67% looks low but is expected. The semantic similarity metric (distilUSE) also uses a tokenizer that treats Cyrillic characters as unknown, so it cannot recognise that `"discоvered"` means `"discovered"`. The *human-perceived* similarity is effectively 100% — you cannot see the difference.

---

## DeepWordBug Deep Dive — All 5 Scoring Methods × All 5 Transformers

### The 5 Scoring Methods

Scoring answers the question: **which words should I corrupt first?**

---

#### `combined` (default — best in paper)

Averages two temporal scores: THS and TTS. Looks at each word from both the left side and the right side of the sentence.

```
Combined(i) = (THS(i) + TTS(i)) / 2
```

**How it works:** calls both THS and TTS internally — each runs n model queries (one per prefix/suffix), so combined costs **2n queries total** for an n-word sentence.

**Weakness:** does NOT skip stopwords — no explicit filter exists. "the", "and", "is" can rank highly if they happen to appear at confidence inflection points in the prefix/suffix sequence. This can waste your perturbation budget on useless words (as you saw with "the" and "and" getting changed instead of sentiment words).

**Best for:** general use when you don't know which words matter.

---

#### `temporal` (THS — Temporal Head Score)

Reads the sentence **left to right**. Asks: *"how much did adding this word increase model confidence, given everything before it?"*

```
THS(i) = P(label | words[0..i]) - P(label | words[0..i-1])
```

Practical example:

```
Sentence: "the performances were dreadful and exhausting"

Prefix confidences (NEGATIVE), scores relative to uniform baseline (0.50 for 2-class model):
  "the"                              → 0.52   THS = 0.52 - 0.50 = +0.02  (word 0: vs. baseline)
  "the performances"                → 0.53   THS = 0.53 - 0.52 = +0.01
  "the performances were"           → 0.55   THS = 0.55 - 0.53 = +0.02
  "the performances were dreadful"  → 0.91   THS = 0.91 - 0.55 = +0.36  ← spike here
  "the performances were dreadful and" → 0.90   THS = 0.90 - 0.91 = -0.01
  "...dreadful and exhausting"      → 0.97   THS = 0.97 - 0.90 = +0.07

Top word by THS: "dreadful" (biggest left-to-right jump)
```

**Best for:** sentences where the key word appears early and drives everything after it.

---

#### `tail` (TTS — Temporal Tail Score)

Reads the sentence **right to left**. Asks: *"how much does this word contribute when I read from the end?"*

```
TTS(i) = P(label | words[i..n]) - P(label | words[i+1..n])
```

Practical example:

```
Sentence: "the performances were dreadful and exhausting"

Suffix confidences (NEGATIVE), last word uses baseline (0.50) instead of empty suffix:
  "exhausting"                          → 0.85   TTS(last) = 0.85 - 0.50 = +0.35  (vs. baseline)
  "and exhausting"                      → 0.87   TTS = 0.87 - 0.85 = +0.02
  "dreadful and exhausting"             → 0.96   TTS = 0.96 - 0.87 = +0.09
  "were dreadful and exhausting"        → 0.96   TTS = 0.96 - 0.96 = +0.00
  "performances were dreadful..."       → 0.95   TTS = 0.95 - 0.96 = -0.01
  "the performances were dreadful..."   → 0.97   TTS = 0.97 - 0.95 = +0.02

Top word by TTS: "exhausting" (largest tail contribution)
```

**Best for:** sentences where the key sentiment word appears at the end (common in English: "the film was truly **terrible**").

---

#### `replaceone` (Replace-1)

The most direct and reliable method. Replaces each word with `[UNK]` one at a time and measures the exact confidence drop.

```
importance(i) = P(label | original_text) - P(label | text_with_word_i_as_UNK)
```

Practical example:

```
Original: "the performances were dreadful and exhausting"
P(NEGATIVE | original) = 0.97

Replace "the" with [UNK]:        P = 0.97   drop = 0.00
Replace "performances" with [UNK]: P = 0.91   drop = 0.06
Replace "were" with [UNK]:        P = 0.96   drop = 0.01
Replace "dreadful" with [UNK]:    P = 0.61   drop = 0.36  ← biggest drop
Replace "and" with [UNK]:         P = 0.97   drop = 0.00
Replace "exhausting" with [UNK]:  P = 0.82   drop = 0.15

Ranking: dreadful (0.36) > exhausting (0.15) > performances (0.06) > ...
```

**Why use this over combined?** It is completely direct — it measures exactly what happens when each word is masked. Stopwords like "the" and "and" score near zero naturally because replacing them with `[UNK]` barely changes model confidence. There is no explicit stopword filter in the code, but the effect is the same: stopwords sink to the bottom of the ranking on their own.

**Note:** `replaceone` replaces the word text inline as `[UNK]` (the literal string) rather than the tokenizer's UNK ID — the word boundary is preserved but the token becomes unknown to the model.

**Weakness:** costs n model queries (one per word), same as THS or TTS alone. Produces more reliable rankings for short sentences and sentences with one dominant sentiment word.

**Best for:** when you want the attack to focus only on words that actually drive the prediction. Recommended for short sentences.

---

#### `random`

Shuffles the word order randomly. No model queries needed.

```
Ranking: randomly assigned
```

**When to use:** as a baseline to demonstrate that scoring actually matters — if random scoring works as well as combined, the scoring step isn't contributing much. In practice, random is significantly worse.

---

### Scoring Method Comparison on the Same Sentence

```
Sentence: "i went to the theatre and my experience was bad"
(10 words, only "bad" carries sentiment)

combined  → ranks: the(2), and(3), bad(1)  ← wastes 2/3 budget on stopwords
temporal  → ranks: the(2), theatre(3), bad(1)  ← similar problem
tail      → ranks: bad(1), experience(2), my(3)  ← slightly better
replaceone → ranks: bad(1), experience(2), theatre(3)  ← correct, no stopwords

Winner for short sentences with few sentiment words: replaceone
```

---

### The 5 Transformers

Transformers answer: **how exactly do I corrupt the chosen word?**

---

#### `homoglyph` (default — most effective in paper)

Replaces one random character with a **Unicode visual twin** — a character from a different script that looks identical.

```python
HOMOGLYPHS = {
    'a': 'ɑ',   # Latin a  → IPA alpha     (looks identical)
    'e': 'е',   # Latin e  → Cyrillic е    (looks identical)
    'o': 'о',   # Latin o  → Cyrillic о    (looks identical)
    'p': 'р',   # Latin p  → Cyrillic р    (looks identical)
    'c': 'ϲ',   # Latin c  → Greek lunate sigma
    'i': 'і',   # Latin i  → Cyrillic і
    'x': '×',   # Latin x  → multiplication sign
    ...
}
```

Example:
```
"dreadful" → pick random position, say index 2 (character 'e')
           → replace 'e' with 'е' (Cyrillic)
           → "drеadful"

To a human:    "drеadful" looks identical to "dreadful"
To BERT:       different Unicode code point → different or unknown token
```

**Why it's the most effective:** the resulting word looks 100% identical to a human but produces a completely different token sequence. BERT has never been trained on Cyrillic characters embedded in English words.

**Limitation:** only works on characters that have a homoglyph in the table. Words like "bad" (3 chars, only 'a' has a homoglyph) are barely affected because only 1/3 characters can be replaced.

---

#### `swap` — adjacent character swap

Swaps two adjacent characters at a random position.

Code: `word[:i] + word[i+1] + word[i] + word[i+2:]`

```
"dreadful"
 d r e a d f u l
     ↑↑ pick random position i=2 (characters 'e' and 'a')
     swap: put word[3] before word[2]
→ "draedful"
```

Another example, position i=4 (characters 'd' and 'f'):
```
"dreadful"
 d r e a d f u l
         ↑↑ i=4
→ "dreafdful"... wait, let's be precise:
   word[:4]  = "drea"
   word[5]   = "f"        ← i+1
   word[4]   = "d"        ← i
   word[6:]  = "ul"       ← i+2 onward
→ "dreafdul"
```

**Looks like:** a common typo (transposition error).  
**Works because:** the transposed word is not in BERT's vocabulary, splits into unfamiliar subwords.  
**Visible to humans:** yes, slightly — transpositions are noticeable in short words.

---

#### `flip` — random character substitution

Replaces one character with a different random letter. The code picks from `a–y` (25 values via `randint(0,24)+97`), then adds 1 if the result would equal the original character — this effectively gives the full `a–z` range minus the original letter.

```
"dreadful"
 → pick position 5 (character 'f', ASCII 102)
 → pick random int 0–24, add 97 → say result is 107 ('k')
 → 107 >= 102 ('f'), so add 1 → 108 ('l')
 → "dreadlul"
```

Another example where no adjustment needed:
```
"dreadful"
 → pick position 3 (character 'a', ASCII 97)
 → pick random int 0–24, add 97 → say result is 110 ('n')
 → 110 >= 97, so add 1 → 111 ('o')
 → "dreodful"
```

**Looks like:** a typo where you pressed the wrong key.  
**Works because:** the corrupted word is completely unknown to the tokenizer.  
**Visible to humans:** yes, it's clearly a wrong letter.  
**Note:** the replacement character is always lowercase (a–z), even if the original was uppercase.

---

#### `remove` — delete one character

Deletes one random character entirely.

```
"dreadful"
 → pick position 2 (character 'e')
 → delete it
 → "dradful"
```

**Looks like:** a fast-typing deletion error.  
**Works because:** "dradful" is an unknown token.  
**Visible to humans:** usually noticeable, especially in short words.  
**Dangerous edge case:** very short words (3 chars) lose a significant fraction of signal. `"bad"` can become `"ad"`, `"bd"`, or `"ba"` — any of the three characters can be removed.

---

#### `insert` — insert a random character

Inserts one random letter (a–z) at a random position. The insertion point can be anywhere from before the first character to after the last (`randint(0, len(word))`).

```
"dreadful"
 → pick position 4 (between 'd' and 'f')
 → pick random letter: chr(97 + randint(0,25)) → say 'x'
 → "dreadxful"
```

**Looks like:** an accidental keypress.  
**Works because:** "dreadxful" is an unknown token.  
**Visible to humans:** yes, the extra letter is noticeable.  
**Effect on length:** makes the word one character longer, which can cause unfamiliar subword splits.

---

### Transformer Comparison on the Same Word

```
Word: "dreadful"

homoglyph → "drеadful"   looks identical to human, completely different to tokenizer ← best stealth
swap      → "dredaful"   transposition, noticeable but natural-looking typo
flip      → "dreadkul"   wrong letter, slightly unnatural
remove    → "dradful"    missing letter, noticeable
insert    → "dreadxful"  extra letter, noticeable
```

**Stealth ranking:** homoglyph > swap > remove > insert > flip  
**Effectiveness ranking:** homoglyph > swap = remove > insert = flip  
**Works on short words (3 chars):** swap and remove work better than homoglyph (more proportional disruption)

---

### Quick Decision Guide: Which Combination to Use

| Situation | Scoring | Transformer |
|-----------|---------|-------------|
| General use, long sentence | `combined` | `homoglyph` |
| Short sentence, avoid wasting budget on stopwords | `replaceone` | `homoglyph` |
| Sentence where key word is at the end | `tail` | `homoglyph` |
| Sentence where key word is early | `temporal` | `homoglyph` |
| You want realistic-looking typos | `replaceone` | `swap` |
| Baseline / demo of random vs smart scoring | `random` | `homoglyph` |
| Target word is 3 characters or fewer | `replaceone` | `swap` or `remove` |
| Maximum stealth | `replaceone` | `homoglyph` |
| Maximum disruption (don't care about naturalness) | `combined` | `flip` |

---

### Example 1b — TextBugger (Hybrid Character + Word)

**The concept:** Score word importance, then for each important word try all 5 perturbation types (space insert, char delete, char swap, homoglyph/keyboard typo, embedding neighbour) and pick the one that drops confidence the most.

**Settings in app:**
```
Attack:               TextBugger
Model:                [SST-2] BERT Sentiment (textattack)
max_perturbations:    5
mode:                 black-box
strategy:             combined
similarity_threshold: 0.75
```

**Input text to use:**
```
This is one of the worst movies I have ever seen in my life
```

**What the attack does:**

```
Step 1 — Score word importance (sentence + word importance, black-box):
  "worst"  → highest importance  ← drives the NEGATIVE prediction

Step 2 — Try all 5 perturbation types on "worst":
  Insert space:   "wor st"        → query model → confidence drops
  Delete char:    "orst"          → query model → confidence drops
  Swap adjacent:  "owrst"         → query model → confidence drops
  Sub-C (homoglyph+typo): "ԝorst" → query model → confidence drops
  Sub-W (GloVe neighbour): "bad"  → query model → confidence drops

  Pick the one with the maximum confidence drop:
  "wor st" (space insert) causes the largest drop  ← chosen

Step 3 — Apply and check label flip:
  "This is one of the wor st movies I have ever seen in my life"
  Model: LABEL_1 / POSITIVE 99.4%  ← LABEL FLIP → return immediately
```

```
Result:
  Original:    LABEL_0 / NEGATIVE  99.92%
  Adversarial: LABEL_1 / POSITIVE  99.39%  ✓ label flipped

  Adversarial text: "This is one of the wor st movies I have ever seen in my life"

  Perturbation ratio:  71.4%  (set to 5 but only 1 word needed)
  Semantic similarity: 91.3%  (only one word was split with a space)
  Queries:             10     (extremely fast)
  Time:                1.5s   (model already cached)
```

> **Why "wor st" works:** Inserting a space splits "worst" into two tokens: ["wor", "##st"] → the model never learned associations for the fragment "wor". The strong negative signal of "worst" disappears because neither fragment exists as a meaningful token. The rest of the sentence ("I have ever seen in my life") is neutral context — without "worst" as the anchor, the model flips.

---

### Example 2 — Pruthi2019 (Realistic Typos)

**The concept:** Simulate the typos a human makes on a keyboard. Substitute letters with adjacent keys on QWERTY.

**Settings in app:**
```
Attack:           Pruthi2019
Model:            [SST-2] BERT Sentiment (textattack)
max_perturbations: 1
```

**Input text to use:**
```
The acting was phenomenal and the story was deeply moving
```

**What the attack does:**

```
QWERTY keyboard layout:
  q w e r t y u i o p
  a s d f g h j k l
  z x c v b n m

"phenomenal":
  pick character 'e' at position 3
  keyboard neighbors of 'e': w, r, s, d
  substitute → "phwnomenal" or "phrnomenal"
  
Output: "The acting was phwnomenal and the story was deeply moving"
```

**Good for demonstrating:** how fragile models are to realistic, human-like typos. This is especially relevant for autocorrect bypass or SEO spam scenarios.

**Confirmed live result (reproduced in SarabCraft):**

```
Settings:  max_perturbations=2
Input:     The acting was phenomenal and the story was deeply moving
```

```
Pruthi2019 typo operations applied:
  "phenomenal" → insert extra 'o' at position 5    → "phenoomenal"
  "moving"     → substitute 'v' with adjacent key  → 'f' (v and f are adjacent on QWERTY)
                                                   → "mofing"

Adversarial: "The acting was phenoomenal and the story was deeply mofing"
```

```
Result:
  Original:    LABEL_1 / POSITIVE  99.96%
  Adversarial: LABEL_0 / NEGATIVE  99.88%  ✓ label flipped

  Perturbation ratio:  20%   (2 of 10 words changed)
  Semantic similarity: 90.0% (typos only slightly confuse the encoder)
  Queries:             745   (Pruthi tries many typo combinations)
  Time:                77s   (first run, model loading included)
```

> **Key observation:** "phenoomenal" and "mofing" are both obviously wrong to a human — they look like fast-typing errors. But BERT has never seen these token sequences in training. The model loses two of its strongest signal words simultaneously and flips the label despite having been 99.96% confident.

---

### Example 3 — TextFooler (Word-Level, Black-Box)

**The concept:** Replace the most important words with synonyms that pass three filters: word-level cosine similarity, POS matching, and full-sentence semantic similarity.

**Settings in app:**
```
Attack:              TextFooler
Model:               [SST-2] BERT Sentiment (textattack)
max_candidates:      50
similarity_threshold: 0.84
embedding_cos_threshold: 0.5
max_perturbation_ratio: 0.3
```

**Input text to use:**
```
The performances are extraordinary and the direction is masterful
```

**What the attack does, step by step:**

```
Step 1 — Delete-one importance scores:
  "extraordinary" → 0.38  ← removing this word hurts confidence most
  "masterful"     → 0.24
  "performances"  → 0.11

Step 2 — Get embedding neighbours for "extraordinary":
  Counter-fitted embeddings (semantic space):
  nearby words: [remarkable, exceptional, outstanding, spectacular, breathtaking]
  Filtered by cosine ≥ 0.5: all pass

Step 3 — POS filter (strict):
  "extraordinary" is ADJ
  only keep ADJ candidates: [remarkable, exceptional, outstanding, spectacular]

Step 4 — Semantic similarity filter (sentence-level, distilUSE):
  "The performances are remarkable and the direction is masterful"
  sim("original", "candidate") = 0.91  ✓  passes threshold 0.84

Step 5 — Query model:
  "The performances are remarkable and the direction is masterful"
  Model: NEGATIVE (57%)  ← label flipped!  → RETURN immediately

Output: "The performances are remarkable and the direction is masterful"
         original label: POSITIVE 94%  →  adversarial label: NEGATIVE 57%
```

**Key insight:** "remarkable" and "extraordinary" mean almost the same thing to a human. But to BERT, they are *different vectors* in embedding space — BERT was fine-tuned on SST-2 in a way that gave "extraordinary" strong positive weight. Replacing it breaks that learned association.

**Confirmed live result (reproduced in SarabCraft):**

```
Settings:  max_candidates=50  similarity_threshold=0.75  embedding_cos_threshold=0.4
           max_perturbation_ratio=0.5
Input:     This movie was terrible and completely unwatchable from start to finish
```

```
Step 1 — Delete-one importance scores:
  "terrible"     → highest  ← drives the NEGATIVE prediction
  "unwatchable"  → second
  (stopwords filtered out: "was", "and", "from")

Step 2 — Embedding neighbours for "terrible":
  Counter-fitted GloVe space: [awful, dreadful, horrible, atrocious, horrendous]
  Cosine ≥ 0.4: all pass
  POS filter (ADJ): all pass
  Semantic sim check: candidates near 0.78  ✓  passes 0.75 threshold

Step 3 — Query model:
  Try "awful":      still NEGATIVE  → continue
  Try "original":   POSITIVE 99.9%  ← LABEL FLIP → return immediately
  (TextFooler found that "original" — semantically unrelated to "terrible"
   but in a similar embedding cluster — breaks the model's association)

Also changed: "unwatchable" → "faithful"
```

```
Result:
  Original:    LABEL_0 / NEGATIVE  99.92%
  Adversarial: LABEL_1 / POSITIVE  99.92%  ✓ label flipped

  Adversarial text: "This movie was original and completely faithful from start to finish"

  Perturbation ratio:  18%   (2 of 11 words changed — only 1 actually needed)
  Semantic similarity: 78.7% (sentence meaning changed a lot — "original" ≠ "terrible")
  Queries:             20    (very fast — found a flip on second candidate)
  Time:                6s    (model already loaded)
```

> **What this reveals about the model:** BERT-SST2 associated "terrible" with NEGATIVE so strongly that replacing it with almost any non-negative word is enough to flip 99.9% confidence. The word "original" is not inherently positive but its *absence of negative signal* is enough. This is a learned shortcut, not true semantic understanding.

---

### Example 4 — BERT-Attack (Word-Level, MLM-based)

**The concept:** Use BERT itself to suggest substitutions. Feed the *original unmasked text* to the MLM and read predictions at all positions in one shot.

**Settings in app:**
```
Attack:                 BERT-Attack
Model:                  [SST-2] BERT Sentiment (textattack)
max_candidates:         48
max_perturbation_ratio: 0.4
use_bpe:                true
```

**Input text to use:**
```
An absolute disaster of a film with no redeeming qualities whatsoever
```

**What the attack does, step by step:**

```
Step 1 — [UNK] importance scoring:
  Replace each word with [UNK], measure confidence drop
  "disaster"   → 0.41  ← most important
  "redeeming"  → 0.29
  "absolute"   → 0.18

Step 2 — MLM forward pass on ORIGINAL text (the innovation):
  Feed "An absolute disaster of a film with no redeeming qualities whatsoever"
  BERT predicts top-48 tokens at EVERY position simultaneously
  
  At position 2 ("disaster"):
    top predictions: ["catastrophe", "failure", "mistake", "tragedy", "mess", ...]
    These are contextually appropriate because BERT sees the full sentence

Step 3 — Filter candidates (no inline similarity check in BERT-Attack):
  Remove stopwords and subword fragments (##)
  Keep: ["catastrophe", "failure", "mistake", "tragedy", "mess"]

Step 4 — Try each, pick best confidence drop:
  "An absolute catastrophe of a film..." → POSITIVE 54%  ← LABEL FLIP → return

Output: "An absolute catastrophe of a film with no redeeming qualities whatsoever"
```

**Compare to TextFooler:**  
TextFooler uses a *static* word vector space (counter-fitted GloVe). BERT-Attack uses the *model's own context* to find substitutions — better candidates because they fit the surrounding sentence.

**Confirmed live result (reproduced in SarabCraft):**

```
Settings:  max_candidates=48  max_perturbation_ratio=0.4
           threshold_pred_score=0.0  use_bpe=true
Input:     An absolute disaster of a film with no redeeming qualities whatsoever
```

```
Step 1 — [UNK] importance scoring:
  "disaster"  → largest confidence drop when masked  ← target first

Step 2 — MLM forward pass on ORIGINAL (unmasked) text:
  BERT reads the full sentence and predicts top-48 tokens at every position
  At "disaster" position, top contextual predictions include: success, triumph...
  (BERT knows the sentence context and suggests words that fit the slot)

Step 3 — Try "success":
  "An absolute success of a film with no redeeming qualities whatsoever"
  Model: LABEL_1 / POSITIVE 64.8%  ← LABEL FLIP → return immediately
```

```
Result:
  Original:    LABEL_0 / NEGATIVE  99.93%
  Adversarial: LABEL_1 / POSITIVE  64.8%  ✓ label flipped

  Adversarial text: "An absolute success of a film with no redeeming qualities whatsoever"

  Perturbation ratio:   9.1%  (1 of 11 words changed — single word flip)
  Semantic similarity: 83.7%  (sentence meaning changed significantly)
  Queries:             29     (fast — UNK scoring + MLM pass done in one shot)
  Time:                57s    (first run, model loading included)
```

> **Why this is impressive:** changing only ONE word ("disaster" → "success") flipped a 99.93% confident NEGATIVE to POSITIVE. The rest of the sentence — "no redeeming qualities whatsoever" — is still clearly negative, but without the word "disaster" as the anchor, BERT's classifier lost its grip. The MLM chose "success" because it is the most grammatically and contextually fitting antonym for "disaster" in that slot.

---

### Example 5 — PWWS (WordNet Synonyms)

**The concept:** Use classic WordNet (the linguistic synonym database), but weight words using a combined saliency × confidence drop formula.

**Settings in app:**
```
Attack:          PWWS
Model:           [SST-2] BERT Sentiment (textattack)
max_candidates:  10
```

**Input text to use:**
```
A beautiful and heartwarming story about love and family
```

**What the attack does:**

```
PWWS score formula:
  H(word) = ΔP(y|x) × softmax(S(word))

  Where:
    ΔP  = drop in predicted-class confidence when word is deleted
    S   = word saliency score
    The product means: a word must BOTH matter to the model AND have usable synonyms

Step 1 — WordNet synonyms for "beautiful":
  Synsets: [beautiful → lovely, gorgeous, stunning, attractive, pretty]
  POS filter: only keep ADJ replacements (Penn Treebank tag: JJ)

Step 2 — H-scores (combined importance × softmax):
  "beautiful"    → H = 0.31
  "heartwarming" → H = 0.24
  "love"         → H = 0.09

Step 3 — Greedy substitution in descending H order:
  Try "lovely" for "beautiful":
    "A lovely and heartwarming story about love and family"
    Model: NEGATIVE 52%  ← label flip! → done

Output: "A lovely and heartwarming story about love and family"
```

> **Important note on WordNet synsets in SarabCraft:** PWWS relies on WordNet's synonym database. WordNet sometimes maps common words to unexpected synsets — "movie" maps to "moving-picture show", "film" maps to "moving-picture show", "acting" maps to "playacting". These archaic synonyms often fail to fool the model because they are too unusual. PWWS works best on sentences with common adjectives that have clean, natural synonyms (e.g., "beautiful" → "lovely", "awful" → "dreadful").

---

### Example 6 — Alzantot GA (Genetic Algorithm)

**The concept:** Run an evolutionary algorithm. Maintain a population of 60 mutated versions of the text, breed them together, and evolve toward texts that fool the model.

**Settings in app:**
```
Attack:            Alzantot GA
Model:             [SST-2] BERT Sentiment (textattack)
population_size:   60
max_generations:   20
mutation_rate:     1.0
similarity_threshold: 0.8
```

**Input text to use:**
```
This is one of the greatest films I have ever seen in my entire life
```

**What the attack does (generation by generation):**

```
Initial population (60 texts, each with 1 word mutated):
  Individual 1:  "This is one of the greatest films I have ever seen in my complete life"
  Individual 2:  "This is one of the greatest films I have ever witnessed in my entire life"
  Individual 3:  "This is one of the best films I have ever seen in my entire life"
  ...
  Individual 60: "This is one of the greatest pictures I have ever seen in my entire life"

Fitness function (untargeted):
  fitness = 1.0 - P(POSITIVE | text)
  We want to MINIMIZE confidence in POSITIVE

Generation 1 scores:
  Individual 3  ("best" instead of "greatest"):  fitness = 0.22
  Individual 15 ("finest" instead of "greatest"): fitness = 0.19
  Individual 7  ("greatest" + "witnessed"):       fitness = 0.31  ← best

Parent selection (softmax, temperature=0.3):
  Higher fitness → more likely to be selected as parent
  But NOT deterministic — even bad individuals have small chance

Crossover (uniform, 50% per word):
  Parent A: "This is one of the greatest pictures I have ever seen..."
  Parent B: "This is one of the worst films I have ever witnessed..."
  Child:    "This is one of the worst pictures I have ever witnessed..."
              ↑ word-by-word coin flip between parents

Mutation (best-improvement):
  For child, try all synonyms at one word position
  Pick the substitution that maximally increases fitness

...20 generations later...

Best individual: "This is one of the most dismal films I have ever observed"
Model: NEGATIVE (71%)  ← success
```

**What makes GA powerful:** it doesn't commit to one word at a time. It explores many *combinations* of substitutions simultaneously, finding multi-word changes that work together in ways a greedy approach would miss.

---

### Example 7 — HotFlip (White-Box, Gradient)

**The concept:** This is the only attack that opens the model's weights. It computes the gradient of the loss with respect to each token's embedding vector, then finds the single token swap across the entire vocabulary that causes the maximum damage — using math, not queries.

**Settings in app:**
```
Attack:             HotFlip
Model:              [SST-2] BERT Sentiment (textattack)
max_flips:          5
beam_width:         10
max_perturbed:      2
similarity_threshold: 0.8
```

**Input text to use:**
```
The film is a stunning achievement in modern cinema
```

**What the attack does:**

```
Step 1 — Forward pass + gradient:
  Feed text to BERT, compute loss for predicted class (POSITIVE)
  Backpropagate to get gradient at each token's embedding:
  
  token "stunning" has gradient vector: [0.41, -0.22, 0.88, ...]  (768 numbers)
  token "achievement" has gradient: [0.12, 0.09, -0.31, ...]

Step 2 — Taylor approximation score for every possible swap:
  For each (position, new_token) pair, estimate the effect on loss:
  
  score(pos="stunning", new_token="mediocre") 
    = (embed("mediocre") - embed("stunning")) · gradient("stunning")
    ≈ 0.62  ← large positive = this swap hurts the model's confidence a lot

  score(pos="stunning", new_token="boring") 
    ≈ 0.58

  Best swap globally: "stunning" → "mediocre"

Step 3 — Beam search (width=10):
  Keep 10 candidate texts, expand each by one more swap, score all
  
  Beam after flip 1:
    "The film is a mediocre achievement in modern cinema"  score=0.71
    "The film is a dull achievement in modern cinema"      score=0.68
    ...

Step 4 — Check constraints on each candidate:
  WordEmbeddingDistance: cos("stunning", "mediocre") ≥ 0.8?  → check
  PartOfSpeech: both ADJ? → yes ✓

Step 5 — Check label flip:
  "The film is a mediocre achievement in modern cinema"
  Model: NEGATIVE (73%)  ← success at flip 1

Output: "The film is a mediocre achievement in modern cinema"
```

**Why it's different from all others:** it never tries random candidates. The gradient tells it *exactly* which swap will hurt the model most, using a first-order Taylor approximation. It's much faster per query than black-box attacks.

---

### Example 8 — Back-Translation (Sentence-Level)

**The concept:** Translate the sentence into another language and back. The translation process naturally paraphrases — same meaning, different words and structure.

**Settings in app:**
```
Attack:                   Back-Translation
Model:                    [SST-2] BERT Sentiment (textattack)
num_paraphrases:          5
similarity_threshold:     0.6
chained_back_translation: 0
target_lang:              es
```

**Input text to use:**
```
The screenplay is weak and the characters are completely underdeveloped
```

**What the attack does:**

```
Round-trip translation (English → Spanish → English):

  EN: "The screenplay is weak and the characters are completely underdeveloped"
   ↓ MarianMT (opus-mt-en-ROMANCE)  [English → Spanish]
  ES: "El guión es débil y los personajes están completamente subdesarrollados"
   ↓ MarianMT (opus-mt-ROMANCE-en)  [Spanish → English]
  EN: "The script is feeble and the characters are totally underdeveloped"

Semantic similarity check:
  sim("The screenplay is weak...", "The script is feeble...") = 0.81  ✓

Model query:
  "The script is feeble and the characters are totally underdeveloped"
  Model: POSITIVE (54%)  ← label flip!  (confused by "feeble" instead of "weak")

Output: "The script is feeble and the characters are totally underdeveloped"
```

**Try chained back-translation (more aggressive):**  
Set `chained_back_translation: 3` — it routes through 3 random pivot languages (e.g. EN → FR → EN → IT → EN → RO → EN). Each hop introduces more paraphrasing, making it harder for the model to recognize the original signal words.

> **Note on first-run behaviour:** Back-Translation requires the MarianMT ROMANCE models (`Helsinki-NLP/opus-mt-en-ROMANCE` and `opus-mt-ROMANCE-en`) to be downloaded on first use. The download takes 1–2 minutes. After that, the models are cached and translations are fast. If you see `perturbation_ratio: 0.0` and identical input/output, the model is still loading — wait and retry.

---

## Part 4 — Attack Comparison

| Attack | Speed | Stealth | Access needed | Best for |
|--------|-------|---------|---------------|----------|
| DeepWordBug | ⚡⚡⚡ Fast | Medium | Black-box | Tokenizer-blind models |
| Pruthi2019 | ⚡⚡⚡ Fast | High | Black-box | Realistic typo simulation |
| TextFooler | ⚡⚡ Medium | High | Black-box | General-purpose word substitution |
| BERT-Attack | ⚡⚡ Medium | High | Black-box | Contextually fluent substitutions |
| PWWS | ⚡⚡ Medium | High | Black-box | Linguistically grounded (WordNet) |
| Alzantot GA | ⚡ Slow | Very High | Black-box | Hard targets, multi-word changes |
| HotFlip | ⚡⚡⚡ Fast | Medium | **White-box** | Research, gradient access available |
| Back-Translation | ⚡ Slow | Very High | Black-box | Structurally different paraphrases |
| Bad Characters | ⚡⚡ Medium | **Imperceptible** | Black-box | Visually-identical Unicode attacks (homoglyph/invisible/deletion/reordering) |

### Model Query Count (for a 12-word sentence)

```
DeepWordBug (combined): ~24 queries  (n prefix queries + n suffix queries, n=12 words)
DeepWordBug (replaceone): ~12 queries (one [UNK] query per word)
Pruthi2019:     ~0 queries  (no model queries needed — purely character-level)
TextFooler:   ~200 queries  (delete-one scoring + candidate evaluation per word)
BERT-Attack:  ~150 queries  (UNK scoring + candidate evaluation)
PWWS:          ~60 queries  (saliency scoring + substitution)
Alzantot GA: ~1200 queries  (60 population × 20 generations)
HotFlip:        ~20 queries  (beam search, gradient does the heavy lifting)
Back-Translation: ~5 queries (one per paraphrase candidate)
```

---

## Part 5 — What to Try Next

### Changing the model

Switch from sentiment (SST-2) to topic classification:

```
Model: [AG News] BERT Topic (textattack)
Labels: World / Sports / Business / Sci/Tech

Input: "Scientists have discovered a new exoplanet with signs of water"
Expected: Sci/Tech

Attack with TextFooler → see if it can force the model to classify as Sports
```

### Targeted attacks

In the app, specify a `target_label` to force the model to predict a *specific* class:

```
Text:         "The acting was wooden and the plot made no sense"
Original:     NEGATIVE
Target label: POSITIVE
Attack:       TextFooler

The attack will now search specifically for substitutions that push toward POSITIVE,
not just any label change.
```

### Transfer attacks

Run an attack on one model and check if the adversarial text also fools a *different* model without re-running. This tests transferability — how general the vulnerability is.

```
Source model:  [SST-2] BERT Sentiment (textattack)
Target model:  [SST-2] DistilBERT Sentiment
                [Yelp]  BERT Sentiment

Run TextFooler on source model → use "Verify Transfer" to test on targets
```

### Experiment: same text, all attacks

Use this text on every attack and compare results:

```
The film is a complete disappointment and a waste of everyone's time
```

Compare:
- Which attack succeeds first (fewest perturbations)?
- Which attack produces the most human-readable adversarial text?
- Which attack has the highest semantic similarity in the result?
- Which attack uses the fewest model queries?

---

## Quick Reference — App Settings for All Attacks

| Attack | Key params to change | Expected effect |
|--------|---------------------|-----------------|
| DeepWordBug | `transformer=swap` vs `homoglyph` | homoglyph works better, swap more visible |
| DeepWordBug | `max_perturbations=1` vs `10` | 1 = minimal change, may fail; 10 = higher success |
| TextFooler | `similarity_threshold=0.9` | stricter = more human-like, may fail more |
| TextFooler | `similarity_threshold=0.5` | looser = higher success, less natural |
| BERT-Attack | `max_candidates=10` vs `48` | fewer candidates = faster but weaker |
| Alzantot GA | `population_size=10, max_generations=5` | fast demo, lower success rate |
| HotFlip | `beam_width=1` | greedy instead of beam search |
| Back-Translation | `target_lang=fr` | French pivot instead of Spanish |
| Back-Translation | `chained_back_translation=3` | stronger paraphrase via 3 pivots |

---

## Part 6 — Confirmed Live Results

All run on `[SST-2] BERT Sentiment (textattack)`. LABEL_0 = NEGATIVE, LABEL_1 = POSITIVE.

---

**DeepWordBug** | `scoring=replaceone` `transformer=homoglyph` `max_perturbations=3`
```
Scientists have discovered a new exoplanet with signs of water
→ Scientists have disϲovered a ոew exoplanet with signs оf water
LABEL_1 88.3% → LABEL_0 98.4% ✓
```

---

**DeepWordBug** | `scoring=combined` `transformer=homoglyph` `max_perturbations=3`
```
Scientists have discovered a new exoplanet with signs of water
→ Scieոtists have diѕcovered a new exoplanet with signs of ԝater
LABEL_1 88.3% → LABEL_0 98.9% ✓
```

---

**DeepWordBug** | `scoring=combined` `transformer=flip` `max_perturbations=3`
```
Scientists have discovered a new exoplanet with signs of water
→ Scienthsts have discoveced a new exoplanet with signs of qater
LABEL_1 88.3% → LABEL_0 90.1% ✓
```

---

**DeepWordBug** | `scoring=replaceone` `transformer=homoglyph` `max_perturbations=3`
```
The story had some interesting moments but the pacing was quite slow
→ The story hɑd some interesting moments Ьut the pacing was quite ѕlow
LABEL_0 99.6% → LABEL_1 72.2% ✓
```

---

**DeepWordBug** | `scoring=combined` `transformer=homoglyph` `max_perturbations=3`
```
A few scenes were genuinely funny but overall the film felt forgettable
→ A few sϲenes werе genuinely funny but overall the film felt 𝚏orgettable
LABEL_0 99.6% → LABEL_1 96.9% ✓
```

---

**DeepWordBug** | `scoring=combined` `transformer=homoglyph` `max_perturbations=3`
```
The performances were lackluster and the storyline was completely predictabe
→ The performances weⲅe lac𝒌luster and the storyline was completely prediϲtabe
LABEL_0 99.9% → LABEL_1 73.5% ✓
```

---

**DeepWordBug** | `scoring=replaceone` `transformer=homoglyph` `max_perturbations=1`
```
i went to the theatre and my experience was bad
→ i went to the theatre and my experience was baԁ
LABEL_0 99.9% → LABEL_1 63.2% ✓
```
