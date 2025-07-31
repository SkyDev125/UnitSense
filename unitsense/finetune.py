import json
import os
import sys
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitsense.constants import BIO_LABELS, MODEL, FINETUNED_MODEL

# get tagged sentences from file
tagged_sentences_path = "./data/tagged_sentences.json"

tokenized_sentences = []
labeled_sentences = []
if os.path.exists(tagged_sentences_path):
    with open(tagged_sentences_path, encoding="utf-8") as f:
        tagged_data = json.load(f)
        tagged_sentences = tagged_data.get("tagged_sentences", [])
        tokenized_sentences = [
            sentence.get("tokens", []) for sentence in tagged_sentences
        ]
        labeled_sentences = [
            sentence.get("labels", []) for sentence in tagged_sentences
        ]

if not tokenized_sentences or not labeled_sentences:
    raise ValueError("No tokenized sentences or labels found in tagged_sentences.json")

# Map the labels to their corresponding indices
label_to_index = {label: index for index, label in enumerate(BIO_LABELS)}


# Convert labels to indices
def convert_labels_to_indices(labels):
    return [label_to_index[label] for label in labels]


indexed_sentences = []
max_tokens = 0
for labeled_sentence in labeled_sentences:
    indices = convert_labels_to_indices(labeled_sentence)
    indexed_sentences.append(indices)
    max_tokens = max(max_tokens, len(indices))


# Pad sentences to the maximum length
def add_padding(arr, length, mode):
    if mode == "tokens":
        return arr + ["[PAD]"] * (length - len(arr)) if len(arr) < length else arr
    elif mode == "labels":
        return arr + [-100] * (length - len(arr)) if len(arr) < length else arr


padded_tokenized_sentences = [
    add_padding(sentence, max_tokens, "tokens") for sentence in tokenized_sentences
]

padded_indexed_sentences = [
    add_padding(sentence, max_tokens, "labels") for sentence in indexed_sentences
]

# Load models needed
model = AutoModel.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# Convert to input Ids
def convert_to_input_ids(tokenized_sentences):
    input_ids = []
    for sentence in tokenized_sentences:
        ids = [
            tokenizer.vocab.get(token, tokenizer.vocab["[UNK]"]) for token in sentence
        ]
        input_ids.append(ids)
    return input_ids


padded_input_ids_sentences = convert_to_input_ids(padded_tokenized_sentences)


# Create attention masks
def create_attention_mask(input_ids_batch):
    pad_token_id = tokenizer.pad_token_id
    return [
        [0 if token_id == pad_token_id else 1 for token_id in sentence]
        for sentence in input_ids_batch
    ]


attention_masks = create_attention_mask(padded_input_ids_sentences)

# Convert to tensors
input_ids_tensor = torch.tensor(padded_input_ids_sentences)
attention_masks_tensor = torch.tensor(attention_masks)
labels_tensor = torch.tensor(padded_indexed_sentences)


# Create dataset
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


dataset = NERDataset(
    input_ids=input_ids_tensor,
    attention_masks=attention_masks_tensor,
    labels=labels_tensor,
)

# Create DataLoader
batch_size = 8
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the model for token classification
model_to_train = AutoModelForTokenClassification.from_pretrained(
    MODEL, num_labels=len(BIO_LABELS)
)

# Create optimizer
optimizer = torch.optim.AdamW(model_to_train.parameters(), lr=5e-5)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_to_train.to(device)


# Training loop
num_epochs = 20  # or however many you want
model_to_train.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model_to_train(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model_to_train.save_pretrained(FINETUNED_MODEL)
tokenizer.save_pretrained(FINETUNED_MODEL)
