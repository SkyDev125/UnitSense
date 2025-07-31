import json
import os
import sys
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitsense.constants import BIO_LABELS, FINETUNED_MODEL

# Load the finetuned model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(FINETUNED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
model.eval()

# Load test data
with open("./data/tagged_sentences.json", encoding="utf-8") as f:
    test_data = json.load(f)["tagged_sentences"]

# Test the model on the test data
for sentence in test_data:
    tokens = sentence["tokens"]
    labels = sentence["labels"]

    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids_tensor = torch.tensor([input_ids])
    attention_mask = torch.tensor([[1] * len(input_ids)])

    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Map predictions to label names
    pred_labels = [BIO_LABELS[p] for p in predictions]

    print("Tokens:", tokens)
    print("True Labels:", labels)
    print("Predicted Labels:", pred_labels)
    print("---")
