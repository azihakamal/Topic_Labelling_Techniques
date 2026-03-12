# -*- coding: utf-8 -*-
from pathlib import Path
import re
import unicodedata
import pandas as pd
import numpy as np
import umap
import spacy
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# =============== CONFIG ===============
INPUT_DIR = Path("./label_article/newspaper_articles/output_articles_csv_clean")
OUTPUT_DIR = Path("./label_article/newspaper_articles/output_article_labeling_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MIN_DF = 5
MAX_DF = 0.95
N_NEIGHBORS = 15
N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 12
HDBSCAN_MIN_SAMPLES = 2
MAX_DOC_CHARS = 80000

# =============== STOPWORDS ===============
nltk.download("stopwords", quiet=True)
german_stopwords = stopwords.words("german")
english_stopwords = stopwords.words("english")

custom_stopwords = [
    "universität", "uni", "hochschulen", "studierende", "pnn", "million",
    "professor", "studenten", "studium", "mehr", "siehe", "pressemitteilung",
    "jahr", "jahren", "prozent", "link", "springer", "prof", "com", "milliarde",
    "seite", "abbildung", "foto", "bild", "lesen", "artikel", "quelle", "euro",
    "mensch", "problem", "kind", "frage", "platz", "geld", "mann", "professur",
    "dpa", "oderzeitung", "barnim", "echo", "oranienburger", "name", "markisch",  
    "netz", "alloemeine", "auf", "meter", "letzter", "abstand", "monat", "tag", 
    "info", "standard", "", "hellip", "furs", "dafur", "gunth", "auflage",
    "kilometer", "hand", "leute", "funf", "jury", "tweet", "prozent", "jahr", 
    "hochschule", "universitat", "student", "semester","tausend", "allgemein", 
    "stunde", "jahrhundert", "studierend", "studentin", "woche",
    "fachhochschule", "hochschule",
    "wintersemester", "sommersemester", "erstsemester",
    "programm", "prasident",
    "form", "person"
]

MONTH_NAMES = {
    "januar", "februar", "märz", "april", "mai", "juni", "juli",
    "august", "september", "oktober", "november", "dezember",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}

all_stopwords = list(set(german_stopwords + english_stopwords + custom_stopwords))
all_stopwords_norm = all_stopwords

# =============== CLEANING ===============
def clean_social_noise(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return text.strip()

def norm_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = clean_social_noise(t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.casefold()
    t = re.sub(r"\b[kq]?witter\b", "twitter", t)
    t = re.sub(r"\b\d{2,4}\b", "", t)
    return t.strip()

def identity_func(x):
    return x

# =============== SPACY MODELS ===============
print(" Loading spaCy models...")
nlp_de = spacy.load("de_core_news_md", disable=["parser", "textcat"])
nlp_multi = spacy.load("xx_ent_wiki_sm")
nlp_de.max_length = 2_000_000
extended_stopwords = set(all_stopwords_norm)

# =============== HELPER FUNCTIONS ===============
def split_long_text(text, max_chars=MAX_DOC_CHARS):
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

VERB_STOPWORDS = {"sagen", "machen", "geben", "kommen", "gehen",
                  "fallen", "bleiben", "liegen", "stehen", "sehen",
                  "zeigen", "finden", "wissen", "denken", "bringen",
                  "nehmen", "arbeiten"}

def should_keep_token(token, detected_entities):
    if token.text.lower() in detected_entities:
        return False
    if token.text.lower() in MONTH_NAMES:
        return False
    if token.ent_type_ in {"PER","DATE","TIME","MONEY","PERCENT","QUANTITY"}:
        return False
    if token.is_stop or token.is_punct or token.like_num or token.is_currency or token.is_space:
        return False
    if token.pos_ not in {"NOUN","ADJ","VERB"}:
        return False
    lemma = token.lemma_.casefold()
    if len(lemma) < 3 or lemma in extended_stopwords or lemma in VERB_STOPWORDS:
        return False
    return True


def spacy_filter_text(text):
    if len(text) > MAX_DOC_CHARS:
        print(f" Splitting long article ({len(text)} chars)")
        parts = split_long_text(text)
        processed_parts = [spacy_filter_text(p) for p in parts]
        return " ".join(processed_parts)
    doc_multi = nlp_multi(text)
    detected_entities = {ent.text.lower() for ent in doc_multi.ents}
    doc = nlp_de(text)
    tokens = [token.lemma_.casefold() for token in doc if should_keep_token(token, detected_entities)]
    return " ".join(tokens)

# =============== LOAD & FILTER ARTICLES ===============
all_texts_raw, all_texts_filtered, file_map, dfs = [], [], [], {}
dropped_records = []
total_articles = 0

for csv_file in INPUT_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)
    if "clean_article" not in df.columns:
        print(f"Skipping {csv_file}, no 'clean_article' column found.")
        continue
    dfs[csv_file.name] = df.copy()
    for idx, text in enumerate(df["clean_article"].dropna().tolist()):
        total_articles += 1
        cleaned = norm_text(str(text))
        filtered = spacy_filter_text(cleaned)

        if not filtered.strip():
            filtered = cleaned
            note = "fallback_to_raw"
        elif len(filtered.split()) < 5:
            filtered = cleaned
            note = "short_fallback"
        else:
            note = "ok"

        all_texts_raw.append(cleaned)
        all_texts_filtered.append(filtered)
        file_map.append((csv_file.name, idx))

        if note != "ok":
            dropped_records.append({"source_file": csv_file.name, "row_id": idx, "note": note})
            print(f" {note} → {csv_file.name}:{idx}")

print(f" Total articles found: {total_articles}")
print(f" Articles processed for embedding: {len(all_texts_raw)} (should match total)")

if dropped_records:
    dropped_df = pd.DataFrame(dropped_records)
    drop_path = OUTPUT_DIR / "dropped_articles_log.csv"
    dropped_df.to_csv(drop_path, index=False)
    print(f" Logged {len(dropped_records)} fallback articles → {drop_path}")
else:
    print(" No fallback articles required.")

# =============== EMBEDDINGS ===============
print(" Generating Sentence-BERT embeddings from raw text...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = embedding_model.encode(all_texts_raw, show_progress_bar=True, convert_to_numpy=True)

# =============== VECTORIZER ===============
vectorizer_model = CountVectorizer(
    preprocessor=identity_func,
    stop_words=all_stopwords_norm,
    lowercase=False,
    ngram_range=(1, 3),
    min_df=MIN_DF,
    max_df=MAX_DF
)

# =============== UMAP + HDBSCAN ===============
umap_model = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    n_components=N_COMPONENTS,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# =============== REPRESENTATION MODELS (KeyBERT + MMR) ===============
rep_kbi = KeyBERTInspired()
rep_mmr = MaximalMarginalRelevance(diversity=0.5)

# =============== TRAIN BERTOPIC ===============
print(" Training BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    representation_model=rep_kbi,
    language="multilingual",
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(all_texts_filtered, embeddings)
topics = np.asarray(topics)
final_topics = topics.copy()

# =============== HANDLE NOISE (INITIAL REASSIGNMENT) ===============
noise_idx = np.where(topics == -1)[0]
if noise_idx.size > 0:
    print(f" Found {noise_idx.size} noise docs → reassigning to nearest topics...")
    clustered_idx = np.where(topics != -1)[0]
    if clustered_idx.size > 0:
        clustered_embeddings = embeddings[clustered_idx]
        clustered_topics = topics[clustered_idx]
        for i in noise_idx:
            sims = cosine_similarity(
                embeddings[i].reshape(1, -1),
                clustered_embeddings
            )[0]
            final_topics[i] = clustered_topics[np.argmax(sims)]
else:
    print(" No noise docs found.")

# =============== UPDATE & REDUCE TOPICS ===============
print(" Refining topics with MMR for better diversity...")
topic_model.update_topics(
    all_texts_filtered,
    vectorizer_model=vectorizer_model,
    representation_model=rep_mmr
)
topic_model = topic_model.reduce_topics(all_texts_filtered, nr_topics="auto")

# =============== RECOMPUTE TOPICS AFTER REDUCTION ===============
print(" Recomputing topic assignments after reduction...")
new_topics, new_probs = topic_model.transform(all_texts_filtered)
final_topics = np.array(new_topics)

# =============== FINAL REASSIGNMENT TO REMOVE -1 ===============
noise_idx = np.where(final_topics == -1)[0]
if noise_idx.size > 0:
    print(f" Final pass: {noise_idx.size} unclustered docs → reassigning to nearest topics...")
    clustered_idx = np.where(final_topics != -1)[0]
    clustered_embeddings = embeddings[clustered_idx]
    clustered_topics = final_topics[clustered_idx]
    for i in noise_idx:
        sims = cosine_similarity(
            embeddings[i].reshape(1, -1),
            clustered_embeddings
        )[0]
        final_topics[i] = clustered_topics[np.argmax(sims)]
else:
    print(" All docs assigned after reduction.")

# =============== GENERATE LABELS ===============
print(" Generating final topic labels...")
new_labels = topic_model.generate_topic_labels(
    nr_words=3,
    separator=" / ",
    topic_prefix=False,
    word_length=3
)
topic_model.set_topic_labels(new_labels)

topics_dict = topic_model.get_topics()
topic_labels = [
    " / ".join([w for w, _ in topics_dict[t][:3]]) if t in topics_dict and len(topics_dict[t]) > 0 else "Other"
    for t in final_topics
]

# =============== SAVE OUTPUTS ===============
results = pd.DataFrame({
    "source_file": [f for f, _ in file_map],
    "row_id": [i for _, i in file_map],
    "article_raw": all_texts_raw,
    "article_filtered": all_texts_filtered,
    "topic_id": final_topics,
    "topic_label": topic_labels
})
combined_path = OUTPUT_DIR / "all_articles_with_topics.csv"
results.to_csv(combined_path, index=False)
print(f" Saved → {combined_path}")

topic_info = topic_model.get_topic_info()
summary_path = OUTPUT_DIR / "topic_summary.csv"
topic_info.to_csv(summary_path, index=False)
print(f" Saved → {summary_path}")

model_dir = OUTPUT_DIR / "bertopic_model"
topic_model.save(model_dir)
print(f" Model saved → {model_dir}/")

# =============== WRITE PER-FILE CSVS ===============
print("✍ Writing per-file topic CSVs...")
for (fname, idx), (tid, tlabel) in zip(file_map, zip(final_topics, topic_labels)):
    dfs[fname].loc[idx, "topic_id"] = int(tid)
    dfs[fname].loc[idx, "topic_label"] = tlabel

for fname, df in dfs.items():
    out_path = OUTPUT_DIR / fname
    df.to_csv(out_path, index=False)
    print(f"   → {out_path}")

print(" All per-file topic CSVs written.")
print(" Topic labeling with KeyBERT + MMR completed successfully.")
