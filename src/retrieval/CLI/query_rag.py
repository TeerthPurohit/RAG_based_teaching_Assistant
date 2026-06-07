import os
import sys
import numpy as np
import joblib
from pydantic import BaseModel 
from sentence_transformers import SentenceTransformer , CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.config.config_env import api_key
from google import genai

# Load model and embeddings once at startup
client = genai.Client(api_key=api_key)
model = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#embeddings path files and folders
embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/embeddings/embeddings.joblib"))
df = joblib.load(embeddings_path)


class QueryResponse(BaseModel):
    answer: str\

#query expansion 
def expand_query(query_text: str) -> list[str]:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f'Generate 3 alternative search queries for this question in a Machine Learning for Trading course. Return ONLY the queries, one per line: "{query_text}"'
    )
    return [query_text] + response.text.strip().split("\n")[:3]


def rerank(query_text: str, candidates: list, top_k: int = 6) -> list:
    scores = reranker.predict([(query_text, r["text"]) for r in candidates])
    return [c for _, c in sorted(zip(scores, candidates), reverse=True)[:top_k]]

#main query

def query_rag(query_text: str) -> QueryResponse:
    queries = expand_query(query_text)
    embeddings_matrix = np.vstack(df["embedding"])
    
    seen, all_rows = set(), []
    for q in queries:
        sims = cosine_similarity(embeddings_matrix, [model.encode([q])[0]]).flatten()
        for idx in sims.argsort()[::-1][:15]:
            if idx not in seen:
                seen.add(idx)
                all_rows.append(df.iloc[idx].to_dict())

    final_chunks = rerank(query_text, all_rows, top_k=8)

    import json
    context = json.dumps(
        [
            {
                "title": r["title"],
                "number": r["number"],
                "start": r["start"],
                "end": r["end"],
                "text": r["text"],
            }
            for r in final_chunks
        ],
        indent=2,
    )

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
{context}

User Question:
"{query_text}"

Additional rules:
- Mention video number and timestamp naturally (e.g. "Video 196 at 22:00")
- Reference at most 3 videos, only the most relevant
- Be friendly and clear like a human tutor
- If the retrieved context does not contain enough information to answer the question,
  respond ONLY with: "This topic isn't covered in the course materials I have access to."
- Do NOT use outside knowledge to supplement the answer under any circumstances.
'''
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    answer = response.text.strip()

    return QueryResponse(answer=answer)