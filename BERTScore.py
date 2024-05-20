pip install docx bert-score
import docx
from bert_score import score
from tabulate import tabulate

# Function to read text from a DOCX file
def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Path to the DOCX file containing the original text
original_text_path = "/content/drive/MyDrive/Colab Notebooks/datasets/a_letter_to_god.docx"
original_text = read_docx(original_text_path)

# Generated summaries
summaries = {
    'google/flan-t5-base': 'Lencho was an ox of a man, working like an animal in the fields, but still he knew how to write. The following Sunday, at daybreak, he began to write a letter to God.',
    'google/t5-base': 'lencho was an ox of a man, working like an animal in the fields, but still he knew how to write . he wrote a letter to God, asking for money from his employees, he himself gave part of his salary, and several friends of his were obliged to give something ‘for an act of charity’ . when he opened the letter, it was evident that to answer it he needed something more than goodwill,',
    'facebook/bart-large-cnn': 'Lencho, who knew his fields intimately, saw the sky towards the north-east. A strong wind began to blow and along with the rain very large hailstones began to fall. For an hour the hail rained on the house, the garden, the hillside, the cornfield, on the whole valley.',
    'sshleifer/distilbart-cnn-12-6': 'In the heart of all who lived in that solitary house in the middle of the valley, there was a single hope: help from God . The following Sunday, at daybreak, he began to write a letter which he himself would carry The following Sunday Lencho came a bit earlier than usual to ask if there was a letter for him It was the postman himself who handed the letter to him . Lencho showed not the slightest surprise on seeing the money; such was',
    # Add more summaries if needed
}

# List to hold the results
results = []

# Calculate BERTScore for each summary
for model_name, summary in summaries.items():
    P, R, F1 = score([summary], [original_text], lang='en', verbose=True)
    results.append([model_name, f'{P.mean().item():.4f}', f'{R.mean().item():.4f}', f'{F1.mean().item():.4f}'])

# Print the results in a table
headers = ["Model", "BERTScore Precision", "BERTScore Recall", "BERTScore F1"]
print(tabulate(results, headers, tablefmt="grid"))
