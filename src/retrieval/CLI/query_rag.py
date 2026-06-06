import os
import sys
import numpy as np
import joblib
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config.config_env import api_key
from google import genai

# Load model and embeddings once at startup
client = genai.Client(api_key=api_key)

model = SentenceTransformer("BAAI/bge-m3")

embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/embeddings/embeddings.joblib"))
df = joblib.load(embeddings_path)


class QueryResponse(BaseModel):
    answer: str


def query_rag(query_text: str) -> QueryResponse:
    question_embedding = model.encode([query_text])[0]

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
"{query_text}"
Instructions:
- Answer naturally like a human tutor.
- DO NOT use markdown.
- DO NOT use **, *, #, bullets.
- Write in plain clean text.
- Mention video number and timestamp naturally.
- Mention only most relevant videos.
- Maximum 3 recommendations.
'''

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = (
        response.text
        .replace("**", "")
        .replace("* ", "")
        .replace("###", "")
        .replace("##", "")
        .strip()
    )

    return QueryResponse(answer=answer)