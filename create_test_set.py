import pandas as pd
from collections import Counter, defaultdict
import re
from tqdm import tqdm
import math

# ----------- Config -----------
INPUT_CSV = "sample_50_rows.csv"
OUTPUT_CSV = "matching_results.csv"
CHUNKSIZE = 5000
DECIMAL_PLACES = 1  # Number of decimal places for matching scores


# ----------- Helpers -----------
def tokenize(text):
    return re.findall(r"\b[^\W\d_]+\b", str(text).lower())


# ----------- Step 1: Count word frequency across the whole dataset -----------
word_freq = Counter()
total_words = 0

print("Step 1: Calculating word frequencies...")
for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE):
    for col in ["title", "sub_title", "answer"]:
        chunk[col] = chunk[col].fillna("")
        for text in chunk[col]:
            tokens = tokenize(text)
            word_freq.update(tokens)
            total_words += len(tokens)

print(
    f"✅ Total unique words: {len(word_freq)} | Total word occurrences: {total_words}"
)

# ----------- Step 2: Load all data for matching -----------
print("Step 2: Loading all data (title, sub_title, answer)...")
df_full = pd.read_csv(INPUT_CSV, usecols=["title", "sub_title", "answer"])
df_full = df_full.fillna("")
sub_titles = df_full["sub_title"].tolist()
titles = df_full["title"].tolist()
answers = df_full["answer"].tolist()

# Build inverted index: word -> set of row indices
word_to_rows = defaultdict(set)
for idx, (title, sub_title, answer) in enumerate(zip(titles, sub_titles, answers)):
    # Combine all text from title, sub_title, and answer
    combined_text = f"{title} {sub_title} {answer}"
    tokens = set(tokenize(combined_text))
    for token in tokens:
        word_to_rows[token].add(idx)

# ----------- Step 3: Calculate matches and matching scores (Original Logic) -----------
print("Step 3: Finding subtitle-to-subtitle matches...")
results = []

# Find min and max word frequencies for normalization
min_freq = min(word_freq.values())
max_freq = max(word_freq.values())

for i, sub_title in tqdm(
    enumerate(sub_titles), total=len(sub_titles), desc="Subtitle matching"
):
    query_tokens = tokenize(sub_title)
    matched_rows = set()
    scores = defaultdict(float)

    for token in query_tokens:
        if word_freq[token] > 0:  # Safety check
            # Option 1: Normalized inverse frequency (0 to 1 range)
            normalized_score = (max_freq - word_freq[token]) / (max_freq - min_freq)

            # Option 2: Log-normalized score (uncomment to use this instead)
            # log_freq = math.log(word_freq[token])
            # max_log_freq = math.log(max_freq)
            # min_log_freq = math.log(min_freq)
            # normalized_score = (max_log_freq - log_freq) / (max_log_freq - min_log_freq)

            # Option 3: Sigmoid-based normalization (uncomment to use this instead)
            # sigmoid_input = 10 * (1 - word_freq[token] / max_freq)  # Scale factor 10
            # normalized_score = 1 / (1 + math.exp(-sigmoid_input))

            for match_idx in word_to_rows.get(token, []):
                if match_idx == i:
                    continue  # skip self
                scores[match_idx] += normalized_score
                matched_rows.add(match_idx)

    # Normalize final scores to 0-1 range based on query length
    max_possible_score = len(
        set(query_tokens)
    )  # Maximum score if all tokens are rarest

    for match_idx in matched_rows:
        final_score = (
            scores[match_idx] / max_possible_score if max_possible_score > 0 else 0
        )
        results.append(
            {
                "query_sub_title": sub_title,
                "matched_sub_title": sub_titles[match_idx],
                "matching_score": round(final_score, DECIMAL_PLACES),
            }
        )

print(f"✅ Subtitle-to-subtitle matches: {len(results)} rows")

# ----------- Step 4: Add word-to-subtitle matching rows -----------
print("Step 4: Adding word-to-subtitle matches...")

# Calculate base scores for each word (using SAME logic as subtitle matching)
word_base_scores = {}
for word, freq in word_freq.items():
    if max_freq > min_freq:
        # Use the SAME scoring logic as the original subtitle matching
        base_score = (max_freq - freq) / (max_freq - min_freq)
    else:
        base_score = 1.0
    word_base_scores[word] = base_score

# ----------- Step 4: Add word-to-subtitle matching rows with NORMALIZED scoring -----------
print("Step 4: Adding word-to-subtitle matches with normalized scoring...")

# Calculate document frequency for each word (how many documents contain the word)
word_doc_freq = {}
total_documents = len(sub_titles)

for word in word_freq.keys():
    doc_count = 0
    for i in range(len(sub_titles)):
        title = titles[i]
        sub_title = sub_titles[i]
        answer = answers[i]
        combined_text = f"{title} {sub_title} {answer}"
        combined_tokens = set(tokenize(combined_text))

        if word in combined_tokens:
            doc_count += 1

    word_doc_freq[word] = doc_count

# Calculate all TF-IDF scores first to find min/max for normalization
all_tfidf_scores = []
temp_word_doc_scores = []

for word in word_freq.keys():
    # Calculate IDF for this word
    doc_freq = word_doc_freq[word]
    if doc_freq > 0:
        idf = math.log(total_documents / doc_freq)
    else:
        idf = 0

    # Check each document for this word
    for i in range(len(sub_titles)):
        title = titles[i]
        sub_title = sub_titles[i]
        answer = answers[i]
        combined_text = f"{title} {sub_title} {answer}"
        combined_tokens = tokenize(combined_text)

        word_count_in_doc = combined_tokens.count(word)
        total_words_in_doc = len(combined_tokens)

        if total_words_in_doc > 0 and word_count_in_doc > 0:
            tf = word_count_in_doc / total_words_in_doc
            tfidf_score = tf * idf
        else:
            tfidf_score = 0.0

        all_tfidf_scores.append(tfidf_score)
        temp_word_doc_scores.append((word, sub_title, tfidf_score))

# Find normalization parameters (exclude zeros for better scaling)
non_zero_scores = [score for score in all_tfidf_scores if score > 0]
if non_zero_scores:
    min_tfidf = min(non_zero_scores)
    max_tfidf = max(non_zero_scores)
else:
    min_tfidf = max_tfidf = 1.0

print(f"TF-IDF range: {min_tfidf:.6f} to {max_tfidf:.6f}")

# Add normalized word-to-subtitle rows
word_subtitle_count = 0
for word, sub_title, raw_score in tqdm(
    temp_word_doc_scores, desc="Adding normalized word-subtitle matches"
):
    if raw_score == 0.0:
        normalized_score = 0.0
    else:
        # OPTION 1: Min-Max normalization (0.1 to 1.0 range for non-zero scores)
        if max_tfidf > min_tfidf:
            normalized_score = 0.1 + 0.9 * (raw_score - min_tfidf) / (
                max_tfidf - min_tfidf
            )
        else:
            normalized_score = 1.0

        # OPTION 2: Simple binary scoring (uncomment to use instead)
        # normalized_score = 1.0 if raw_score > 0 else 0.0

        # OPTION 3: Frequency-based scoring (uncomment to use instead)
        # word_frequency = word_freq[word]
        # if max_freq > min_freq:
        #     normalized_score = (max_freq - word_frequency) / (max_freq - min_freq)
        # else:
        #     normalized_score = 1.0

    results.append(
        {
            "query_sub_title": word,
            "matched_sub_title": sub_title,
            "matching_score": round(normalized_score, DECIMAL_PLACES),
        }
    )
    word_subtitle_count += 1

print(f"✅ Word-to-subtitle normalized matches added: {word_subtitle_count} rows")

# ----------- Step 5: Save to CSV -----------
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_CSV, index=False)

total_subtitle_matches = len(results) - word_subtitle_count
print(f"\n📊 Final Results Summary:")
print(f"   - Subtitle-to-subtitle matches: {total_subtitle_matches} rows")
print(f"   - Word-to-subtitle matches: {word_subtitle_count} rows")
print(f"   - Total rows in output: {len(result_df)} rows")
print(f"✅ Done! All matching results saved to '{OUTPUT_CSV}'")

# ----------- Optional: Display sample of new word-subtitle rows -----------
print(f"\n🔍 Sample of word-subtitle matches:")
word_rows = result_df[result_df["query_sub_title"].str.len() < 20]  # Show shorter words
print(word_rows.head(10).to_string(index=False))
