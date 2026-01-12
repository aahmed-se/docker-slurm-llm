import os
import json
import urllib.request

# --- Config ---
SHARED_DIR = "/data/data"
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
OUTPUT_FILE = os.path.join(SHARED_DIR, "tinyshakespeare.txt")
META_FILE = os.path.join(SHARED_DIR, "meta.json")

def main():
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR)

    # 1. Download Data
    print(f"Downloading {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, OUTPUT_FILE)
        print(f"Saved raw text to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to download: {e}")
        return

    # 2. Build Vocabulary
    print("Building vocabulary...")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) } # char to int
    itos = { i:ch for i,ch in enumerate(chars) } # int to char

    print(f"Vocab size: {vocab_size}")

    # 3. Save Metadata
    meta = { 'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)
    
    print(f"Saved metadata to: {META_FILE}")

if __name__ == "__main__":
    main()