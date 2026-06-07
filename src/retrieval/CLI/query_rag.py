import os
import sys
import numpy as np
import joblib
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
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
You are a Teaching Assistant for "Machine Learning for Trading" by Professor Tucker Balch.

IMPORTANT - YOU MUST FORMAT YOUR RESPONSE USING MARKDOWN:
- Start with a ## header summarizing the topic
- Use **bold** for every key term (e.g. **long position**, **short selling**)
- Use bullet points (- ) for any list of steps, differences, or concepts
- Use > blockquotes when quoting or paraphrasing what a lecture says
- Use `backticks` for formulas or variable names
- Never write long plain paragraphs — break everything into sections

Relevant transcript chunks:
{new_df[["title","number","start","end","text"]].to_json(orient="records")}

User Question:
"{query_text}"

Additional rules:
- Mention video number and timestamp naturally (e.g. "Video 196 at 22:00")
- Reference at most 3 videos, only the most relevant
- Be friendly and clear like a human tutor
'''
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = response.text.strip()

    return QueryResponse(answer=answer)