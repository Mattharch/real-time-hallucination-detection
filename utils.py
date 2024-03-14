import os
import pandas as pd
import torch
import spacy
from openai import OpenAI

# Load Spacy English model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Initialize device for PyTorch based on MPS availability
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

# Ensure the API key is correctly set
os.environ["OPENAI_API_KEY"] = ""

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def llm_evaluate(sentences, sampled_passages):
    # Construct the detailed prompt for consistency scoring
    prompt = f"""You will be provided with a text passage \
                and your task is to rate the consistency of that text to \
                that of the provided context. Your answer must be only \
                a number between 0.0 and 1.0 rounded to the nearest two \
                decimal places where 0.0 represents no consistency and \
                1.0 represents perfect consistency and similarity. \n\n \
                Text passage: {sentences}. \n\n \
                Context: {sampled_passages[0]} \n\n \
                {sampled_passages[1]} \n\n \
                {sampled_passages[2]}."""
    
    # Send the constructed prompt to the model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            # The following user message is optional and can be adjusted based on your requirements.
            {"role": "user", "content": "Please rate the consistency."}
        ]
    )

    # Extract and return the consistency score from the model's response
    try:
        # Attempt to parse the response as a float
        consistency_score = float(response.choices[0].message.content.strip())
    except ValueError:
        # Default to 0.0 if the response is not a valid float
        consistency_score = 0.0

    return consistency_score


def generate_3_samples(prompt):
    sampled_passages = []
    for _ in range(3):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        # Corrected way to access the message content from the response
        text = response.choices[0].message.content.strip()
        sampled_passages.append(text)
    return sampled_passages



def get_output_and_samples(prompt):
    output = generate_3_samples(prompt)[0]
    sampled_passages = generate_3_samples(prompt)
    return output, sampled_passages

def get_cos_sim(output, sampled_passages):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode([output])
    samples_embeddings = model.encode(sampled_passages)
    cos_sim = pairwise_cos_sim(sentence_embeddings, samples_embeddings)
    cos_sim_mean = cos_sim.mean()
    return round(cos_sim_mean.item(), 2)

def get_bertscore(output, sampled_passages):
    sentences = [sent.text.strip() for sent in nlp(output).sents]
    selfcheck_bertscore = SelfCheckBERTScore()
    sent_scores_bertscore = selfcheck_bertscore.predict(sentences=sentences, sampled_passages=sampled_passages)
    df = pd.DataFrame({
        'Sentence Number': range(1, len(sent_scores_bertscore) + 1),
        'Hallucination Score': sent_scores_bertscore
    })
    return df

def get_self_check_nli(output, sampled_passages):
    sentences = [sent.text.strip() for sent in nlp(output).sents]
    selfcheck_nli = SelfCheckNLI(device=mps_device) # Ensure this device argument matches your setup ('cpu' or 'cuda')
    sent_scores_nli = selfcheck_nli.predict(sentences=sentences, sampled_passages=sampled_passages)
    df = pd.DataFrame({
        'Sentence Number': range(1, len(sent_scores_nli) + 1),
        'Probability of Contradiction': sent_scores_nli
    })
    return df
