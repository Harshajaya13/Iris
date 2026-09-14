import os
import tiktoken
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(CURRENT_DIR, "canary_dataset.txt") 
TRAIN_FILE = os.path.join(CURRENT_DIR, "train.bin")
VAL_FILE = os.path.join(CURRENT_DIR, "val.bin")


if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Source text file not found at: {INPUT_FILE}")


print(f"Reading {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()


print("Encoding tokens with tiktoken (gpt2)...")
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(content)

total_tokens = len(tokens)
print(f"Total tokens encoded: {total_tokens:,}")

if total_tokens == 0:
    raise ValueError("Input file is empty. No tokens generated.")


split_idx = int(total_tokens * 0.9)
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

train_array = np.array(train_tokens, dtype=np.uint16)
val_array = np.array(val_tokens, dtype=np.uint16)


train_array.tofile(TRAIN_FILE)
val_array.tofile(VAL_FILE)


print("\n--- Artifact Summary ---")
print(f"train.bin: {len(train_array):,} tokens | Size: {train_array.nbytes / (1024 * 1024):.2f} MB")
print(f"val.bin:   {len(val_array):,} tokens | Size: {val_array.nbytes / (1024 * 1024):.2f} MB")
print(f"Output directory: {CURRENT_DIR}")