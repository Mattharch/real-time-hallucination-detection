import streamlit as st
import utils  # This module needs to be updated as shown below
from time import time

# Streamlit app layout
st.title('Anti-Hallucination Chatbot')

# Text input
user_input = st.text_input("Enter your text:")

if user_input:
    prompt = user_input
    output, sampled_passages = utils.get_output_and_samples(prompt)

    # LLM score calculation
    start = time()
    self_similarity_score = utils.llm_evaluate(output, sampled_passages)
    
    # Try to convert the score to a float, or extract the first number if it fails
    try:
        self_similarity_score = float(self_similarity_score)
    except ValueError:
        self_similarity_score = 0  # Default to 0 if conversion fails

    end = time()

    # Display the output based on the self-similarity score
    st.write("**LLM output:**")
    if self_similarity_score > 0.5:
        st.write(output)
    else:
        st.write("I'm sorry, but I don't have the specific information required to answer your question accurately.")

    # Optional: Display processing time
    st.write(f"Processed in {end - start:.2f} seconds")
