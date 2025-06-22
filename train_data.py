import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import IterableDataset, DataLoader
import os

# -------- Config --------
CSV_FILE = "gensim_matching_results_with_words.csv"
MODEL_NAME = "sagorsarker/bangla-bert-base"
BATCH_SIZE = 16
EPOCHS = 1  # Start with 1, you can resume training later
OUTPUT_DIR = "output/bangla-sbert-finetuned"
CHUNKSIZE = 100_000  # Load 100k rows at a time
SKIP_HEADER = True
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batch
SAVE_EVERY_STEPS = 10000  # Save checkpoint periodically


# -------- InputExample Stream --------
class SentencePairDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        chunk_iter = pd.read_csv(
            self.file_path,
            chunksize=CHUNKSIZE,
            usecols=["query_sub_title", "matched_sub_title", "matching_score"],
        )
        for chunk in chunk_iter:
            chunk.dropna(inplace=True)
            chunk = chunk[chunk["matching_score"] >= 0]
            for _, row in chunk.iterrows():
                yield InputExample(
                    texts=[str(row["query_sub_title"]), str(row["matched_sub_title"])],
                    label=float(row["matching_score"]),
                )


# -------- Load Base Model --------
print("🚀 Loading model...")
word_embedding_model = models.Transformer(MODEL_NAME)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# -------- Dataset and Dataloader --------
print("🔄 Preparing streaming dataset...")
train_dataset = SentencePairDataset(CSV_FILE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# -------- Loss --------
train_loss = losses.CosineSimilarityLoss(model)

# -------- Train --------
print("🧠 Starting training with streaming and gradient accumulation...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    show_progress_bar=True,
    output_path=OUTPUT_DIR,
    save_best_model=False,
    checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints"),
    checkpoint_save_steps=SAVE_EVERY_STEPS,
    use_amp=True,  # Mixed precision for better memory efficiency
)


print(f"✅ Training complete. Model saved to: {OUTPUT_DIR}")
