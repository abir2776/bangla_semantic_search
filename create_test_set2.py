from gensim import corpora, models, similarities
import pandas as pd
import re
from tqdm import tqdm
import csv
import os

# ----------------------- Config -----------------------
INPUT_CSV = "questions_202506011720.csv"
OUTPUT_CSV = "gensim_matching_results_with_words.csv"
SIMILARITY_THRESHOLD_SUBTITLE = 0.0
SIMILARITY_THRESHOLD_WORD = 0.0
TOP_K = 100  # Limit to top 5 matches per item


# ----------------------- Tokenizer -----------------------
def tokenize(text):
    return re.findall(r"\b[^\W\d_]+\b", str(text).lower())


# ----------------------- Load and Preprocess Data -----------------------
print("🔄 Loading and preprocessing data...")
df = pd.read_csv(INPUT_CSV)
df = df.fillna("")
texts = df["sub_title"].astype(str).tolist()
tokenized_texts = [tokenize(t) for t in texts]

# ----------------------- Build Dictionary and Corpus -----------------------
print("🧠 Building dictionary and TF-IDF corpus...")
dictionary = corpora.Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# ----------------------- Build Disk-Based Similarity Index -----------------------
print("⚡ Building disk-backed similarity index...")
index = similarities.Similarity(
    output_prefix="gensim_similarity_index",
    corpus=corpus_tfidf,
    num_features=len(dictionary),
    chunksize=256,
)

# ----------------------- Open CSV Writer -----------------------
print(f"💾 Writing results to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["query_sub_title", "matched_sub_title", "matching_score"]
    )
    writer.writeheader()

    # -------- Subtitle-to-Subtitle Matching --------
    print("🔍 Subtitle-to-subtitle matching...")
    for i, tfidf_vec in tqdm(enumerate(corpus_tfidf), total=len(corpus_tfidf)):
        sims = list(enumerate(index[tfidf_vec]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        count = 0
        for j, score in sims:
            if i != j and score >= SIMILARITY_THRESHOLD_SUBTITLE:
                writer.writerow(
                    {
                        "query_sub_title": texts[i],
                        "matched_sub_title": texts[j],
                        "matching_score": round(float(score), 1),
                    }
                )
                count += 1
                if count >= TOP_K:
                    break

    # -------- Word-to-Subtitle Matching --------
    print("🔤 Word-to-subtitle matching...")
    all_words = list(dictionary.token2id.keys())
    for word in tqdm(all_words, desc="Processing words"):
        word_bow = dictionary.doc2bow([word])
        if not word_bow:
            continue
        word_tfidf = tfidf[word_bow]
        sims = list(enumerate(index[word_tfidf]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        # count = 0
        for j, score in sims:
            if score >= SIMILARITY_THRESHOLD_WORD:
                writer.writerow(
                    {
                        "query_sub_title": word,
                        "matched_sub_title": texts[j],
                        "matching_score": round(float(score), 1),
                    }
                )
                count += 1
                if count >= TOP_K:
                    break

print("✅ Done!")
