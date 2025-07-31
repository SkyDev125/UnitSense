import json
import os
import sys
from transformers import BigBirdTokenizer

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitsense.constants import MODEL

with open("./data/sentenses.json") as f:
    data = json.load(f)

# Load BigBird tokenizer
tokenizer = BigBirdTokenizer.from_pretrained(MODEL)

sentences = data["training_dataset"]
test_sentences = data["testing_dataset"]
tokenized_sentences = []
tokenized_test_sentences = []

for sentence in sentences:
    tokenized = tokenizer(
        sentence,
        return_tensors="pt",
    )

    # Save tokenized sentence with input_ids, attention_mask, word IDs, and offsets
    tokenized_sentences.append(
        {
            "sentence": sentence,
            "tokens": tokenizer.tokenize(sentence),
        }
    )

for sentence in test_sentences:
    tokenized = tokenizer(
        sentence,
        return_tensors="pt",
    )

    # Save tokenized test sentence with input_ids, attention_mask, word IDs, and offsets
    tokenized_test_sentences.append(
        {
            "sentence": sentence,
            "tokens": tokenizer.tokenize(sentence),
        }
    )

# Write updated tokenized_sentences back to file
with open("./data/tokenized_sentences.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "tokenized_sentences": tokenized_sentences,
            "tokenized_test_sentences": tokenized_test_sentences,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
