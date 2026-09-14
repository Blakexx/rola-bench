"""text8 — canonical char-level long-range LM benchmark (the cleaned/lowercased enwik8 sibling;
Transformer-XL, S4, Mamba et al. report text8 bits-per-char). HF mirror `afmck/text8`: one
contiguous string per split (train 90M / val 5M / test 5M chars), 27-symbol alphabet (space + a-z).

Char-level is a feature here: ~27-token vocab kills the embedding tax (no 50k GPT-2 vocab), so the
model's params are real modeling capacity, and it's a genuinely LONG-RANGE continuous stream — train
at any block_size with real dependencies (unlike TinyStories' short packed stories). Metric is
bits-per-char = eval_loss / ln(2).
"""
DATASET = "afmck/text8"
CHARS = " abcdefghijklmnopqrstuvwxyz"          # text8's exact 27-symbol alphabet (space + a-z)
STOI = {c: i for i, c in enumerate(CHARS)}


def vocab_size():
    return len(CHARS)                           # 27


def text8_dataset(block_size, max_train_blocks=None, val_blocks=2000):
    """Materialize the contiguous char stream into fixed block_size LM blocks (labels=input_ids,
    standard next-char LM; drop the trailing remainder so every block is full → no padding)."""
    from datasets import Dataset, DatasetDict, load_dataset

    def enc(split, maxb):
        txt = load_dataset(DATASET, split=split)[0]["text"]      # one row = the full split string
        ids = [STOI[c] for c in txt if c in STOI]
        n = len(ids) // block_size
        if maxb:
            n = min(n, maxb)
        blocks = [ids[i * block_size:(i + 1) * block_size] for i in range(n)]
        return Dataset.from_dict({"input_ids": blocks,
                                  "attention_mask": [[1] * block_size for _ in blocks],
                                  "labels": [b[:] for b in blocks]})

    return DatasetDict(train=enc("train", max_train_blocks),
                       validation=enc("validation", val_blocks))
