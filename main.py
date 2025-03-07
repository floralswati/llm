import transformers
from transformers import pipeline
from docx import Document

model_cache = {}

# Function to create a summarization pipeline for a given model
def create_summarizer(model_name):
    if model_name not in model_cache:
        try:
            summarizer = pipeline('summarization',
                model=model_name,
                max_length=50,
                num_beams=4,
                early_stopping=True)
            model_cache[model_name] = summarizer
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            return None
    return model_cache.get(model_name)


# Function to extract text from a docx file
def extract_text_fromdocx(filename):
    document = Document(filename)
    return [p.text for p in document.paragraphs]

# Function to generate summaries using a given list of summarizers
def generate_summaries(paragraph_texts, summarizers):
    summaries = {}
    for model_name, summarizer in summarizers.items():
        if summarizer is not None:
            try:
                input_length = max(len(p) for p in paragraph_texts)
                max_length = min(input_length, 100)  # Increase max_length to 100
                summary = summarizer(paragraph_texts, max_length=max_length, num_beams=4, early_stopping=True)
                if isinstance(summary, list):
                    summaries[model_name] = summary[0]['summary_text']
                else:
                    summaries[model_name] = summary['summary_text']
            except Exception as e:
                print(f"Error generating summary for '{model_name}': {e}")
        else:
            print(f"Skipping '{model_name}' (model not loaded)")
    return summaries

def main(filename):
    paragraph_texts = extract_text_fromdocx(filename)
    model_names = [
        #"facebook/bart-large-cnn",
        "google/flan-t5-base",
        #"google-t5/t5-base",
        #"gmicrosoft/Phi-3-mini-128k-instruct",
        #"apple/OpenELM-3B-Instruct"
    ]
    summarizers = {model_name: create_summarizer(model_name) for model_name in model_names}
    summaries = generate_summaries(paragraph_texts, summarizers)
    for model_name, summary in summaries.items():
        print(f"Summary generated by {model_name}:\n\n{summary}\n\n")

if __name__ == "__main__":
    main('a_letter_to_god.docx')
