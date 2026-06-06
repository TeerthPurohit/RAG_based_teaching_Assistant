---
title: RAG Teaching Assistant
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# 🎓 RAG Based Teaching Assistant

This RAG based teaching assistant aids students to quickly navigate course material by taking a natural language query and returning the contextual information with an exact video reference and timestamp.

Built for the 📈 Machine Learning for Trading class by Professor Tucker Balch

---

## ✨ Features

🎥 Convert lecture videos/audio to searchable knowledge  
🔍 Semantic search using embeddings  
📚 Retrieve relevant transcript chunks  
🤖 Generate natural answers using Gemini  
⏱️ Suggest precise video numbers and timestamps  
🔁 Continuous Q&A loop (chat like experience)  
💻 Support for local embedding models  

---

## 🏗️ Project Architecture

```text
User Query
 ↓
Embeddings are generated
 ↓
Similarity search on chunked lecture transcripts
 ↓
Top relevant chunks are identified
 ↓
Prompt is constructed and passed to Gemini
 ↓
Gemini generates a natural, helpful answer
 ↓
Recommended video & exact timestamp are displayed
```

---

## 📂 Project Structure

```text
RAGBasedTeaching_Assistant/

 README.md
 requirements.txt
 .gitignore
 .env
 .env.example

 data/
 audios/
 vids/
 transcripts/
 chunks/
 embeddings/
 embeddings.joblib

 models/
 localbgem3/

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
 process_incoming.py

 speech/
 stt.py

 prompts/
 outputs/
 docs/
```

---

## ⚙️ Tech Stack

🐍 Python  
🧠 Sentence Transformers  
📌 BGE-M3 Embeddings  
✨ Google Gemini API  
📊 Scikit-learn  
🐼 Pandas  
💾 Joblib  

---

## 🛠️ Installation

Clone repository:

```bash
git clone YOURREPOLINK
cd RAGBasedTeaching_Assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up .env:

```env
GEMINIAPIKEY=yourapikey
```

---

## ▶️ Run

Start the assistant:

```bash
python src/retrieval/process_incoming.py
```

Example:

```text
Enter your query:
> How is reinforcement learning used in trading?

Thinking...

You'll find this in video 198 between 0:00 and 0:22.

This section explains how trading can be modeled as a reinforcement learning problem where the market acts as the environment.

Continue to video 197 from 3:34 onward for actions and rewards.
```

---

## 🧠 How Retrieval Works

1. ❓ User submits a question  
2. 🔢 Query is embedded  
3. 🔍 Similarity search is performed on lecture transcript database  
4. 📄 Top N most similar chunks are retrieved  
5. 🧩 Relevant chunks and query are passed to Gemini  
6. 🤖 Gemini generates a natural language response  

---

## 🚀 Future Improvements

- 🎨 Streamlit UI  
- 🧠 Conversation memory  
- ⚡ Hybrid retrieval (BM25 + Vector Search)  
- 📚 Multi-course support  
- 🐳 Docker deployment  
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

## 💡 Example Query

```text
How does machine learning help in trading?
```

Output:

```text
Video 164 (0:00-0:21)

This section introduces how hedge funds use machine learning models to predict prices and automate decision making.
```

---

## 👨‍💻 Author

This project was created to explore:

- 🧠 Retrieval-Augmented Generation (RAG)  
- 🔢 Embeddings  
- 🔍 Semantic Search  
- 🤖 LLM-powered tutoring systems