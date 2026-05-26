"""
Build embeddings from JSON chunks using BAAI/bge-m3 model.
Saves embeddings to joblib file for similarity search.
"""
import os
import json
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def build_embeddings(json_dir=None, output_file=None):
    """
    Process JSON files and create embeddings.
    
    Args:
        json_dir: Directory containing JSON files
        output_file: Output path for embeddings joblib file
    """
    if json_dir is None:
        json_dir = os.path.join(os.path.dirname(__file__), "../../data/merged_chunks")
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "../../data/embeddings/embeddings.joblib")
    model = SentenceTransformer("BAAI/bge-m3")
    model.save(os.path.join(os.path.dirname(__file__), "../../models/local_bge_m3"))
    
    my_dicts = []
    chunk_id = 0
    
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(json_dir, json_file), encoding="utf-8") as f:
                content = json.load(f)
            print(f"Creating Embeddings for {json_file}")
            
            texts = [c['text'] for c in content['chunks']]
            embeddings = model.encode(texts)
            
            for i, chunk in enumerate(content['chunks']):
                chunk['chunk_id'] = chunk_id
                chunk['embedding'] = embeddings[i].tolist()
                chunk_id += 1
                my_dicts.append(chunk)
    
    df = pd.DataFrame.from_records(my_dicts)
    joblib.dump(df, output_file)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    build_embeddings()
