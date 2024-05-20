pip install torch
pip install rouge
pip install python-docx rouge-score
import os
from docx import Document
from rouge_score import rouge_scorer

# Function to read text from DOCX file
def read_docx(file_path):
    doc = Document('/content/drive/MyDrive/Colab Notebooks/datasets/a_letter_to_god.docx')
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Function to calculate ROUGE scores
def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Read the original text from the DOCX file
original_text_path = "/content/drive/MyDrive/Colab Notebooks/datasets/a_letter_to_god.docx"
original_text = read_docx(original_text_path)

# Summarization models and their summaries
summaries = {
    'google/flan-t5-base': 'Lencho was an ox of a man, working like an animal in the fields, but still he knew how to write. The following Sunday, at daybreak, he began to write a letter to God.',
    'google/t5-base': 'lencho was an ox of a man, working like an animal in the fields, but still he knew how to write . he wrote a letter to God, asking for money from his employees, he himself gave part of his salary, and several friends of his were obliged to give something ‘for an act of charity’ . when he opened the letter, it was evident that to answer it he needed something more than goodwill,',
    'facebook/bart-large-cnn': 'Lencho, who knew his fields intimately, saw the sky towards the north-east. A strong wind began to blow and along with the rain very large hailstones began to fall. For an hour the hail rained on the house, the garden, the hillside, the cornfield, on the whole valley.',
    'sshleifer/distilbart-cnn-12-6': 'In the heart of all who lived in that solitary house in the middle of the valley, there was a single hope: help from God . The following Sunday, at daybreak, he began to write a letter which he himself would carry The following Sunday Lencho came a bit earlier than usual to ask if there was a letter for him It was the postman himself who handed the letter to him . Lencho showed not the slightest surprise on seeing the money; such was',
    # Add more summaries if needed
}

# Evaluate each summary and print the ROUGE scores
for model_name, summary_text in summaries.items():
    scores = calculate_rouge(original_text, summary_text)
    print(f"Summary by Model: {model_name}")
    print(f"ROUGE-1: {scores['rouge1']}")
    print(f"ROUGE-2: {scores['rouge2']}")
    print(f"ROUGE-L: {scores['rougeL']}\n")
