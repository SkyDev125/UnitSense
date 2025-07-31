import json
from transformers import AutoTokenizer

with open("./data/sentenses.json") as f:
    data = json.load(f)

# Load MiniLM tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

# Sample sentence
sentences = data["sentences"]

tokenized_sentences = []

for sentence in sentences:
    tokenized = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    # Map back to word indices
    word_ids = tokenized.word_ids()

    tokenized_sentences.append(
        {
            "sentence": sentence,
            "tokens": tokenized.tokens(),
            "word_ids": word_ids,
            "offsets": tokenized["offset_mapping"].tolist(),
        }
    )

# Write updated tokenized_sentences back to file
with open("./data/tokenized_sentences.json", "w", encoding="utf-8") as f:
    json.dump(
        {"tokenized_sentences": tokenized_sentences}, f, ensure_ascii=False, indent=2
    )
