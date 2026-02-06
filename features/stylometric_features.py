"""
Stream A — Classical Stylometric Feature Extraction

Extracts a fixed-length numerical vector from a single text block.
In Stage 2, two such vectors are compared via absolute difference
to produce a "dissimilarity vector" for the pairwise classifier.

Feature layout (163 dimensions total):
  [0]       avg_word_length
  [1]       type_token_ratio
  [2]       punctuation_density
  [3]       uppercase_ratio
  [4:10]    POS ratios (NOUN, VERB, ADJ, ADV, PRON, DET)
  [10]      avg_sentence_length
  [11]      flesch_kincaid_grade
  [12]      gunning_fog
  [13:163]  function word frequencies (L2-normalized, 150 words)
"""

import re

import numpy as np
import spacy
import textstat

try:
    # exclude "senter" and "parser" so no component pre-assigns sent_start,
    # then add the fast rule-based sentencizer ourselves
    NLP = spacy.load("en_core_web_sm", exclude=["ner", "parser", "senter"])
    NLP.add_pipe("sentencizer")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    NLP = spacy.load("en_core_web_sm", exclude=["ner", "parser", "senter"])
    NLP.add_pipe("sentencizer")

# ── Top-150 English function words ──
# Function words are largely subconscious style markers — authors don't
# choose prepositions and conjunctions deliberately, making them resistant
# to content-based confounding (unlike nouns or verbs, which vary by topic).
FUNCTION_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "into", "year", "your", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "been", "was", "were", "are", "had", "has", "did",
    "does", "may", "might", "shall", "should", "must", "through", "before",
    "between", "under", "while", "where", "here", "both", "each", "few",
    "more", "such", "own", "same", "too", "very", "during", "whether",
    "however", "although", "nevertheless", "therefore", "thus", "hence",
    "moreover", "furthermore", "meanwhile", "nonetheless", "subsequently",
    "accordingly", "despite", "since", "once", "already", "yet", "still",
    "quite", "rather", "perhaps", "indeed", "certainly", "clearly",
]
FUNCTION_WORDS = list(dict.fromkeys(FUNCTION_WORDS))[:150]
FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

# POS tags to track
POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET"]


# ──────────────────────────────────────────────────────────────
# Feature group 1: Lexical features
# WHY they shift: authors differ in vocabulary richness, punctuation habits,
# and capitalization conventions (e.g., some overuse exclamation marks,
# some capitalize mid-sentence for emphasis).
# ──────────────────────────────────────────────────────────────

def lexical_features(text: str) -> dict:
    tokens = text.split()
    if not tokens:
        return {"avg_word_length": 0.0, "ttr": 0.0,
                "punctuation_density": 0.0, "uppercase_ratio": 0.0}

    word_tokens = [t for t in tokens if t.isalpha()]
    avg_word_length = (
        sum(len(w) for w in word_tokens) / len(word_tokens) if word_tokens else 0.0
    )

    # Type-Token Ratio: unique words / total words.
    # A high TTR indicates varied vocabulary — authors differ in richness.
    ttr = len(set(w.lower() for w in word_tokens)) / len(word_tokens) if word_tokens else 0.0

    punct_count = sum(1 for ch in text if ch in '.,;:!?"\'()-–—')
    punctuation_density = punct_count / len(tokens)

    uppercase_ratio = sum(1 for w in word_tokens if w[0].isupper()) / len(word_tokens) if word_tokens else 0.0

    return {
        "avg_word_length": avg_word_length,
        "ttr": ttr,
        "punctuation_density": punctuation_density,
        "uppercase_ratio": uppercase_ratio,
    }


# ──────────────────────────────────────────────────────────────
# Feature group 2: Syntactic features
# WHY they shift: authors differ in syntactic preferences — e.g., some
# overuse passive constructions (more VERB tags with auxiliary verbs) while
# others prefer noun-heavy academic prose (high NOUN ratio). Average
# sentence length also varies strongly between authors.
# ──────────────────────────────────────────────────────────────

def syntactic_features(text: str) -> dict:
    doc = NLP(text)
    tokens = [t for t in doc if not t.is_space]
    total = len(tokens)

    pos_counts = {tag: 0 for tag in POS_TAGS}
    for token in tokens:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1

    pos_ratios = {f"pos_{tag}": pos_counts[tag] / total if total > 0 else 0.0
                  for tag in POS_TAGS}

    sentences = list(doc.sents)
    avg_sent_len = (
        sum(len([t for t in s if not t.is_space]) for s in sentences) / len(sentences)
        if sentences else 0.0
    )

    return {**pos_ratios, "avg_sentence_length": avg_sent_len}


# ──────────────────────────────────────────────────────────────
# Feature group 3: Function word frequency vector
# L2-normalized so vector magnitude doesn't correlate with block length.
# ──────────────────────────────────────────────────────────────

def function_word_vector(text: str) -> np.ndarray:
    tokens = re.findall(r"\b\w+\b", text.lower())
    vec = np.zeros(len(FUNCTION_WORDS), dtype=np.float32)
    for token in tokens:
        if token in FW_INDEX:
            vec[FW_INDEX[token]] += 1

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ──────────────────────────────────────────────────────────────
# Feature group 4: Readability
# WHY they shift: different authors write at different complexity levels.
# A student who pastes AI-generated text into their essay will often shift
# the reading level noticeably — this is the operational forensics use case.
# ──────────────────────────────────────────────────────────────

def readability_features(text: str) -> dict:
    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
        fog = textstat.gunning_fog(text)
    except Exception:
        fk_grade = fog = 0.0
    return {"flesch_kincaid_grade": fk_grade, "gunning_fog": fog}


# ──────────────────────────────────────────────────────────────
# Master extractor
# ──────────────────────────────────────────────────────────────

def extract_stylometric_features(text: str) -> np.ndarray:
    """
    Returns a ~163-dim float32 feature vector for one text block.
    Feature layout: [lexical(4), syntactic(7), function_words(150), readability(2)]
    """
    lex = lexical_features(text)
    syn = syntactic_features(text)
    fw = function_word_vector(text)
    read = readability_features(text)

    scalar_feats = np.array(
        list(lex.values()) + list(syn.values()) + list(read.values()),
        dtype=np.float32
    )
    return np.concatenate([scalar_feats, fw])


def get_feature_names() -> list[str]:
    """Returns feature names in the same order as extract_stylometric_features."""
    return (
        ["avg_word_length", "ttr", "punctuation_density", "uppercase_ratio"]
        + [f"pos_{t}" for t in POS_TAGS]
        + ["avg_sentence_length"]
        + ["flesch_kincaid_grade", "gunning_fog"]
        + [f"fw_{w}" for w in FUNCTION_WORDS]
    )
