import os, json
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  
import joblib
import requests
from openai import OpenAI

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.config_env import api_key
from google import genai

client = genai.Client(api_key=api_key)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/local_bge_m3"))
model = SentenceTransformer(model_path)
embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/embeddings/embeddings.joblib"))
df = joblib.load(embeddings_path)

def inference_gemini(prompt):
    print("Thinking...")
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return response.text


output_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)


while True:

    input_query = input("\nEnter your query (type exit to quit): ")

    if input_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break


    question_embedding = model.encode([input_query])[0]


    # Find similarities
    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [question_embedding]
    ).flatten()

    top_results = 30
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.iloc[max_indx]


    prompt = f'''
You are a Teaching Assistant for the course "Machine Learning for Trading" by Professor Tucker Balch.

Relevant transcript chunks:
{new_df[["title","number","start","end","text"]].to_json(orient="records")}

----------------
User Question:
"{input_query}"

Instructions:
- Answer naturally like a human tutor.
- DO NOT use markdown.
- DO NOT use **, *, #, bullets.
- Write in plain clean text.
- Mention video number and timestamp naturally.
- Mention only most relevant videos.
- Maximum 3 recommendations.
'''


    with open(os.path.join(output_path, "prompt.txt"), "w") as f:
        f.write(prompt)


    response = inference_gemini(prompt)


    response = (
        response.replace("**", "")
        .replace("* ", "")
        .replace("###", "")
        .replace("##", "")
        .strip()
    )


    print("\n")
    print(response)


    with open(
        os.path.join(output_path, "response.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(response)

