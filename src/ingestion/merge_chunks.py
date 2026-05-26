import os 
import json
import math
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def merge_chunks(input_dir=None, output_dir=None, chunk_size=5):
    if input_dir is None:
        input_dir = os.path.join(os.path.dirname(__file__), "../../data/chunks")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "../../data/merged_chunks")
     
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print(f"Merging chunks in {filename}")
            new_chunks = []
            num_chunks = len(data["chunks"])
            num_groups = math.ceil(num_chunks / chunk_size)
            
            for i in range(num_groups):
                start_index = i * chunk_size
                end_index = min((i + 1) * chunk_size, num_chunks)
                group_chunks = data["chunks"][start_index:end_index]
                
                new_chunks.append({
                    "title": data["chunks"][0]["title"],
                    "number": data["chunks"][0]["number"],
                    "start": group_chunks[0]["start"],
                    "end": group_chunks[-1]["end"],
                    "text": " ".join([chunk["text"] for chunk in group_chunks])
                })
            
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "title": data["chunks"][0]["title"], 
                    "number": data["chunks"][0]["number"], 
                    "chunks": new_chunks
                }, f, ensure_ascii=False, indent=4)
            
            print(f"✓ Merged {filename}")

if __name__ == "__main__":
    merge_chunks()