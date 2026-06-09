---
title: RAG Teaching Assistant
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🎓 RAG Based Teaching Assistant

A RAG-powered chatbot that helps students navigate the **Machine Learning for Trading** course by Professor Tucker Balch (Georgia Tech / Udacity / YouTube).

Ask any question in natural language — get a clear answer with the exact video number and timestamp to watch.

🚀 **Live Demo:** [Chat UI](https://teerthpurohit-rag-tradingwithml-teaching-assistant.hf.space/ui) | [API Docs](https://teerthpurohit-rag-tradingwithml-teaching-assistant.hf.space/docs)

---

## ✨ Features

🎥 Lecture videos converted to a searchable transcript database  
🔍 Semantic search using BGE-M3 embeddings  
🔀 Query expansion — rewrites your question 3 ways for broader retrieval  
🏆 Cross-encoder reranking — scores and selects the most relevant chunks  
🤖 Answer generation using GPT-4o Mini  
⏱️ Returns exact video number and timestamp  
⚡ FastAPI REST backend + Gradio chat UI  
🐳 Deployed on Hugging Face Spaces via Docker  

---

## 🏗️ Architecture

```text
User Query
 ↓
Query Expansion (GPT-4o Mini generates 3 variants)
 ↓
Embedding each variant (BGE-M3)
 ↓
Cosine similarity search → top 15 chunks per variant
 ↓
Deduplicate → ~60 candidate chunks
 ↓
Cross-encoder reranking → top 8 most relevant chunks
 ↓
Prompt constructed and sent to GPT-4o Mini
 ↓
Answer returned with video number and timestamp
```

---

## 📂 Project Structure

```text
RAGBasedTeaching_Assistant/
 README.md
 requirements.txt
 Dockerfile
 app.py
 .env.example
 data/
   transcripts/
   chunks/
   embeddings/
     embeddings.joblib
 models/
   local_bge_m3/
 src/
   config/
     config_env.py
   ingestion/
     mp3totranscripts.py
     mp3tojsons.py
     merge_chunks.py
   embedding/
     embed_builder.py
   retrieval/
     CLI/
       query_rag.py
       process_incoming.py
```

---

## ⚙️ Tech Stack

🐍 Python  
🧠 Sentence Transformers (BGE-M3)  
🏆 CrossEncoder (ms-marco-MiniLM-L-6-v2)  
🤖 OpenAI GPT-4o Mini  
📊 Scikit-learn  
💾 Joblib  
⚡ FastAPI  
🎨 Gradio  
🐳 Docker  

---

## 🛠️ Installation

```bash
git clone YOUR_REPO_LINK
cd RAGBasedTeaching_Assistant
pip install -r requirements.txt
```

Set up `.env`:
```env
OPENAI_API_KEY=your_api_key
```

---

## ▶️ Run

**CLI mode:**
```bash
python src/retrieval/CLI/process_incoming.py
```

**API + UI mode:**
```bash
python app.py
```

- Chat UI: `http://localhost:7860/ui`
- API docs: `http://localhost:7860/docs`

---

## 🧠 How Retrieval Works

1. ❓ User submits a question
2. 🔀 Query is expanded into 3 variants via GPT-4o Mini
3. 🔢 Each variant is embedded with BGE-M3
4. 🔍 Cosine similarity search retrieves top 15 chunks per variant
5. 🧹 Duplicates removed → ~60 unique candidates
6. 🏆 Cross-encoder reranker scores all candidates, keeps top 8
7. 🤖 GPT-4o Mini generates a natural language answer from the best chunks

---

## 🚀 Future Improvements

- 🧠 Conversation memory  
- ⚡ Hybrid retrieval (BM25 + Vector Search)  
- 📚 Multi-course support  
- 🎙️ Voice assistant mode  

---

## 🔐 Environment Variables

Do not upload:
```text
.env
models/
data/embeddings/
```

Upload:
```text
.env.example
```

---

## 👨‍💻 Author

Built by **Teerth Purohit** to explore:
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔀 Query expansion and reranking
- 🔍 Semantic search with vector embeddings
- 🤖 LLM-powered tutoring systems
- 🚀 Cloud deployment on Hugging Face
