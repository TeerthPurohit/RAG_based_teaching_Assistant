import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.CLI.query_rag import QueryResponse, query_rag
import gradio as gr

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


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0f1117 !important;
}

.gradio-container h1 {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #f0f4ff !important;
    letter-spacing: -0.3px;
}

.gradio-container .prose p {
    color: #8b93a8 !important;
    font-size: 0.88rem !important;
}

.message.user {
    background: #1e2130 !important;
    border: 1px solid #2a2f45 !important;
    border-radius: 12px 12px 4px 12px !important;
    color: #dde3f5 !important;
    font-size: 0.92rem !important;
}

.message.bot {
    background: #141824 !important;
    border: 1px solid #1f2538 !important;
    border-radius: 12px 12px 12px 4px !important;
    color: #c8d0e8 !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
}

.gr-textbox textarea, #component-0 textarea {
    font-family: 'DM Sans', sans-serif !important;
    background: #1a1e2e !important;
    border: 1px solid #2a2f45 !important;
    border-radius: 10px !important;
    color: #dde3f5 !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
}

.gr-textbox textarea:focus {
    border-color: #4a6cf7 !important;
    box-shadow: 0 0 0 3px rgba(74, 108, 247, 0.15) !important;
}

#submit-btn, button.primary {
    background: #4a6cf7 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
    color: #fff !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
}

#submit-btn:hover, button.primary:hover {
    background: #3a5ce5 !important;
    transform: translateY(-1px) !important;
}

.examples-holder button {
    background: #1a1e2e !important;
    border: 1px solid #2a2f45 !important;
    border-radius: 8px !important;
    color: #8b93a8 !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s, color 0.2s !important;
}

.examples-holder button:hover {
    border-color: #4a6cf7 !important;
    color: #c8d0e8 !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2f45; border-radius: 4px; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
).set(
    body_background_fill="#0f1117",
    body_text_color="#dde3f5",
    block_background_fill="#141824",
    block_border_color="#1f2538",
    border_color_primary="#2a2f45",
    input_background_fill="#1a1e2e",
    button_primary_background_fill="#4a6cf7",
    button_primary_background_fill_hover="#3a5ce5",
    button_primary_text_color="#ffffff",
)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def ask_question(message: str, history: list) -> str:
    response = query_rag(message)
    return response.answer


chatbot = gr.Chatbot(
    render_markdown=True,
    height=480,
    show_label=False,
    avatar_images=(
        None,
        "https://api.dicebear.com/7.x/bottts/svg?seed=tutor&backgroundColor=4a6cf7",
    ),
)

with gr.Blocks(css=CUSTOM_CSS, theme=THEME) as demo:
    gr.ChatInterface(
        fn=ask_question,
        chatbot=chatbot,
        title="🎓 ML for Trading — Teaching Assistant",
        description=(
            "Ask anything from the **Machine Learning for Trading** course "
            "by Professor Tucker Balch."
        ),
        examples=[
            "What is the difference between shorting and holding a stock?",
            "How does reinforcement learning apply to trading?",
            "What is a Sharpe ratio?",
        ],
    )

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
