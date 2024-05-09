import transformers
from transformers import pipeline
from docx import Document

model_cache = {}

# Function to create a summarization pipeline for a given model
def create_summarizer(model_name):
    if model_name not in model_cache:
        try:
            summarizer = pipeline("text2text-generation", model=model_name, max_length=50, num_beams=4, early_stopping=True)
            model_cache[model_name] = summarizer
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            return None
    return model_cache.get(model_name)

def generate_summaries(paragraph_texts, summarizers):
    summaries = {}
    for model_name, summarizer in summarizers.items():
        if summarizer is not None:
            try:
                summary = summarizer(paragraph_texts, max_length=350, num_beams=4, early_stopping=True)
                if isinstance(summary, list):
                    summaries[model_name] = summary[0]['generated_text']
                else:
                    summaries[model_name] = summary['generated_text']
            except Exception as e:
                print(f"Error generating summary for '{model_name}': {e}")
        else:
            print(f"Skipping '{model_name}' (model not loaded)")
    return summaries

def main(filename):
    document = Document(filename)
    paragraph_texts = [p.text for p in document.paragraphs]
    model_names = [
        #"meta-llama/Llama-2-7b-chat-hf",
        #"google-t5/t5-base",
        #"google-bert/bert-base-uncased",
        #"mistralai/Mixtral-8x7B-Instruct-v0.1",
        "facebook/bart-large-cnn",
        #"openai-community/gpt2-medium",
       # "pegasus-cnn-568M"
    ]

    summarizers = {model_name: create_summarizer(model_name) for model_name in model_names}
    summaries = generate_summaries(paragraph_texts, summarizers)

    for model_name, summary in summaries.items():
        print(f"Summary generated by {model_name}:\n\n{summary}\n\n")

if __name__ == "__main__":
    main('TOP.docx')
