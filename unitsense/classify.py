import json
import sys
import tkinter as tk
from tkinter import messagebox
import os

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unitsense.constants import BIO_LABELS


# Load both datasets from tokenized_sentences.json
with open("./data/tokenized_sentences.json", encoding="utf-8") as f:
    all_data = json.load(f)


# Use correct keys from JSON
DATASET_KEYS = ["tokenized_sentences", "tokenized_test_sentences"]
PRETTY_NAMES = ["Sentences", "Test Sentences"]
current_dataset_key = DATASET_KEYS[0]


def get_dataset():
    return all_data.get(current_dataset_key, [])


def set_dataset_key(new_key):
    global current_dataset_key, current_idx, current_labels, tagged_sentences, tagged_sent_texts
    current_dataset_key = new_key
    # Reload tagged sentences for this dataset from tagged_sentences.json
    tagged_json_path = "./data/tagged_sentences.json"
    if os.path.exists(tagged_json_path):
        with open(tagged_json_path, encoding="utf-8") as f:
            tagged_data = json.load(f)
    else:
        tagged_data = {}
    tagged_key = (
        "tagged_sentences"
        if current_dataset_key == "tokenized_sentences"
        else "tagged_test_sentences"
    )
    tagged_sentences.clear()
    tagged_sent_texts.clear()
    tagged_sentences.extend(tagged_data.get(tagged_key, []))
    tagged_sent_texts.update(ts["sentence"] for ts in tagged_sentences)
    current_idx = 0
    current_labels = []
    update_data()
    next_sentence()


def update_data():
    # Always reload the latest tagged sentences from tagged_sentences.json before filtering
    tagged_json_path = "./data/tagged_sentences.json"
    if os.path.exists(tagged_json_path):
        with open(tagged_json_path, encoding="utf-8") as f:
            tagged_data = json.load(f)
    else:
        tagged_data = {}
    tagged_key = (
        "tagged_sentences"
        if current_dataset_key == "tokenized_sentences"
        else "tagged_test_sentences"
    )
    tagged_sent_texts.clear()
    tagged_sent_texts.update(ts["sentence"] for ts in tagged_data.get(tagged_key, []))
    # Filter out already tagged sentences for the current dataset
    data["tokenized_sentences"] = [
        s for s in get_dataset() if s["sentence"] not in tagged_sent_texts
    ]


# Data structure for the current session
data = {"tokenized_sentences": []}


# Initialize tagged sentences and set for the first dataset
tagged_sentences = []
tagged_sent_texts = set()
update_data()

current_idx = 0
current_labels = []

current_idx = 0
current_labels = []


# Add back the standalone update_status function (move up so it's defined before use)
def update_status():
    if not data["tokenized_sentences"] or current_idx >= len(
        data["tokenized_sentences"]
    ):
        return
    try:
        entry = data["tokenized_sentences"][current_idx]
    except (IndexError, KeyError, TypeError):
        return
    num_tokens = len(entry["tokens"])
    # Update token label backgrounds
    token_labels = [w for w in tokens_frame.winfo_children() if isinstance(w, tk.Label)]
    for i, widget in enumerate(token_labels[:num_tokens]):
        if current_labels[i] is not None:
            widget.config(bg="lightgreen")
        else:
            widget.config(bg="SystemButtonFace")
    # Update label button appearance
    for i in range(num_tokens):
        for j, label in enumerate(BIO_LABELS):
            btn = label_buttons[i][j]
            if current_labels[i] == label:
                btn.config(relief=tk.SUNKEN, bg="#b3d9ff")
            else:
                btn.config(relief=tk.RAISED, bg="SystemButtonFace")
    if all(lab is not None for lab in current_labels):
        next_btn.config(state=tk.NORMAL)
    else:
        next_btn.config(state=tk.DISABLED)


# set_label must also be defined before use
def set_label(idx, label):
    current_labels[idx] = label
    update_status()


# Helper to show 'No sentences left to label' message
def show_no_sentences_message():
    sentence_label.config(
        text="No sentences left to label", font=("Arial", 18, "bold"), fg="gray"
    )
    for widget in tokens_frame.winfo_children():
        widget.destroy()
    next_btn.config(state=tk.DISABLED)


def save_progress():
    # Save tagged sentences for both datasets to a new file tagged_sentences.json
    tagged_json_path = "./data/tagged_sentences.json"
    # Try to load existing tagged data, or start fresh
    if os.path.exists(tagged_json_path):
        with open(tagged_json_path, "r", encoding="utf-8") as f:
            tagged_data = json.load(f)
    else:
        tagged_data = {}
    tagged_key = (
        "tagged_sentences"
        if current_dataset_key == "tokenized_sentences"
        else "tagged_test_sentences"
    )
    tagged_data[tagged_key] = tagged_sentences
    with open(tagged_json_path, "w", encoding="utf-8") as f:
        json.dump(tagged_data, f, ensure_ascii=False, indent=2)


def next_sentence():
    global current_idx, current_labels, label_buttons
    num_sentences = len(data["tokenized_sentences"])
    if num_sentences == 0:
        show_no_sentences_message()
        return
    if current_idx < num_sentences:
        # Save current labels if any
        if current_labels:
            entry = data["tokenized_sentences"][current_idx - 1]
            tagged_sentences.append(
                {
                    "sentence": entry["sentence"],
                    "tokens": entry["tokens"],
                    "labels": current_labels.copy(),
                }
            )
            save_progress()
        if current_idx == num_sentences:
            return  # handled in on_next
        entry = data["tokenized_sentences"][current_idx]
        sentence_label.config(text=f"Sentence: {entry['sentence']}")
        for widget in tokens_frame.winfo_children():
            widget.destroy()
        current_labels = [None] * len(entry["tokens"])
        label_buttons = []  # 2D list: label_buttons[token_idx][label_idx]
        for i, token in enumerate(entry["tokens"]):
            token_label = tk.Label(tokens_frame, text=token, relief="groove", width=15)
            token_label.grid(row=i, column=0, padx=2, pady=2)
            row_buttons = []
            for j, label in enumerate(BIO_LABELS):
                btn = tk.Button(
                    tokens_frame,
                    text=label,
                    width=12,
                    relief=tk.RAISED,
                    bg="SystemButtonFace",
                    command=lambda idx=i, lab=label: set_label(idx, lab),
                )
                btn.grid(row=i, column=j + 1, padx=1, pady=2)
                row_buttons.append(btn)
            label_buttons.append(row_buttons)
        # Change button text to 'Finish' if last sentence, else 'Next Sentence'
        if current_idx == num_sentences - 1:
            next_btn.config(text="Finish")
        else:
            next_btn.config(text="Next Sentence")
        update_status()


def on_next():
    global current_idx
    num_sentences = len(data["tokenized_sentences"])
    if current_idx == num_sentences - 1:
        # Save the last sentence before quitting
        if current_labels:
            entry = data["tokenized_sentences"][current_idx]
            tagged_sentences.append(
                {
                    "sentence": entry["sentence"],
                    "tokens": entry["tokens"],
                    "labels": current_labels.copy(),
                }
            )
            save_progress()
        messagebox.showinfo("Done", "All sentences tagged. Results saved.")
        root.quit()
    else:
        current_idx += 1
        next_sentence()


label_buttons = []  # 2D list for label buttons


root = tk.Tk()
root.title("BIO Tagger")

sentence_label = tk.Label(
    root, text="", font=("Arial", 14), wraplength=700, justify="left"
)
sentence_label.pack(pady=10)

# Scrollable tokens area
canvas = tk.Canvas(root, height=400)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)


# Mouse wheel scrolling support
def _on_mousewheel(event):
    if event.num == 5 or event.delta == -120:
        canvas.yview_scroll(1, "units")
    elif event.num == 4 or event.delta == 120:
        canvas.yview_scroll(-1, "units")


# Windows and MacOS
canvas.bind_all(
    "<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
)
# Linux (event.num 4/5)
canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
scrollbar.pack(side="right", fill="y")

# Replace tokens_frame with scrollable_frame
tokens_frame = scrollable_frame


# Add dataset switch button
def on_switch_dataset():
    idx = DATASET_KEYS.index(current_dataset_key)
    new_idx = (idx + 1) % len(DATASET_KEYS)
    set_dataset_key(DATASET_KEYS[new_idx])
    switch_btn.config(text=f"Switch to {PRETTY_NAMES[idx]}")


switch_btn = tk.Button(
    root, text=f"Switch to {PRETTY_NAMES[1]}", command=on_switch_dataset
)
switch_btn.pack(pady=5)

next_btn = tk.Button(root, text="Next Sentence", command=on_next, state=tk.DISABLED)
next_btn.pack(pady=10)

next_sentence()
root.mainloop()
