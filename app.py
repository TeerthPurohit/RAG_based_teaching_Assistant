import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.CLI.query_rag import QueryResponse, query_rag

app = FastAPI()

class SubmitQueryRequest(BaseModel):
    query_text: str

@app.get("/")
def index():
    return {"message": "RAG Teaching Assistant API"}

@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryResponse:
    query_response = query_rag(request.query_text)
    return query_response
# Gradio UI
def ask_question(message, history):
    response = query_rag(message)
    return response.answer

demo = gr.ChatInterface(
    fn=ask_question,
    title="🎓 ML for Trading Teaching Assistant",
    description="Ask anything from the Machine Learning for Trading course by Professor Tucker Balch.",
    examples=[
        "What is the difference between shorting and holding a stock?",
        "How does reinforcement learning apply to trading?",
        "What is a Sharpe ratio?"
    ],
    theme=gr.themes.Soft()
)

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
