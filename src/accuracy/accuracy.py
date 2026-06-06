import os, json
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from google import genai
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config.config_env import api_key

client = genai.Client(api_key=api_key)

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/local_bge_m3"))
model = SentenceTransformer(model_path)

embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/embeddings/embeddings.joblib"))
df = joblib.load(embeddings_path)


def inference_gemini(prompt):
    print("Thinking...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

eval_data = {
    "question": [],
    "answer": [],
    "contexts": []
}

print("=== RAG Teaching Assistant — Accuracy Evaluation Mode ===")
print("Ask questions, then type 'exit' to see your RAGAS accuracy scores.\n")

while True:
    input_query = input("\nEnter your query (type exit to quit): ")
    if input_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    question_embedding = model.encode([input_query])[0]

    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [question_embedding]
    ).flatten()

    top_results = 30
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.iloc[max_indx]

    # Collect raw chunk texts for RAGAS contexts
    retrieved_chunks = new_df["text"].tolist()

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

    with open(os.path.join(output_path, "response.txt"), "w", encoding="utf-8") as f:
        f.write(response)

    # Store this Q&A for evaluation
    eval_data["question"].append(input_query)
    eval_data["answer"].append(response)
    eval_data["contexts"].append(retrieved_chunks)

    print(f"\n[Collected {len(eval_data['question'])} question(s) for evaluation so far]")

# ---- RAGAS Evaluation after exit ----
if not eval_data["question"]:
    print("No questions collected. Exiting without evaluation.")
else:
    print(f"\n{'='*50}")
    print(f"Running RAGAS evaluation on {len(eval_data['question'])} question(s)...")
    print(f"{'='*50}\n")

    dataset = Dataset.from_dict(eval_data)

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    faithfulness_pct = results['faithfulness'] * 100
    relevancy_pct = results['answer_relevancy'] * 100
    overall_pct = (faithfulness_pct + relevancy_pct) / 2

    print(f"Faithfulness     : {faithfulness_pct:.1f}%  (is the answer grounded in retrieved chunks?)")
    print(f"Answer Relevancy : {relevancy_pct:.1f}%  (does the answer address the question?)")
    print(f"{'─'*50}")
    print(f"Overall Accuracy : {overall_pct:.1f}%")
    print(f"{'='*50}\n")

    # Save results to file
    results_path = os.path.join(output_path, "accuracy_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"RAGAS Evaluation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Questions evaluated : {len(eval_data['question'])}\n\n")
        for i, q in enumerate(eval_data["question"]):
            f.write(f"Q{i+1}: {q}\n")
        f.write(f"\n{'─'*50}\n")
        f.write(f"Faithfulness     : {faithfulness_pct:.1f}%\n")
        f.write(f"Answer Relevancy : {relevancy_pct:.1f}%\n")
        f.write(f"Overall Accuracy : {overall_pct:.1f}%\n")

    print(f"Results saved to: accuracy_results.txt")
