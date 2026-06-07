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


# ── Theme ─────────────────────────────────────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    # Background
    body_background_fill="#0f1117",
    body_background_fill_dark="#0f1117",
    # Panels / blocks
    block_background_fill="#1a1e2e",
    block_background_fill_dark="#1a1e2e",
    block_border_width="1px",
    block_border_color="#2a2f45",
    block_border_color_dark="#2a2f45",
    block_radius="12px",
    # Text
    body_text_color="#c8d0e8",
    body_text_color_dark="#c8d0e8",
    body_text_size="15px",
    # Input
    input_background_fill="#141824",
    input_background_fill_dark="#141824",
    input_border_color="#2a2f45",
    input_border_color_dark="#2a2f45",
    input_border_color_focus="*primary_500",
    input_border_color_focus_dark="*primary_400",
    input_radius="10px",
    # Buttons
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#1a1e2e",
    button_secondary_background_fill_dark="#1a1e2e",
    button_secondary_border_color="#2a2f45",
    button_secondary_text_color="#8b93a8",
    button_large_radius="10px",
    # Chatbot
    chatbot_code_background_color="#0f1117",
    chatbot_code_background_color_dark="#0f1117",
)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def ask_question(message: str, history: list) -> str:
    response = query_rag(message)
    return response.answer


chatbot = gr.Chatbot(
    render_markdown=True,
    height=500,
    show_label=False,
    avatar_images=(
        None,
        "https://api.dicebear.com/7.x/bottts/svg?seed=tutor&backgroundColor=4361ee",
    ),
    placeholder=(
        "<div style='text-align:center; padding: 40px 20px; color: #4a5070'>"
        "<p style='font-size:2rem'>🎓</p>"
        "<p style='font-size:1rem; font-weight:500'>ML for Trading Assistant</p>"
        "<p style='font-size:0.85rem'>Ask anything from Professor Tucker Balch's course</p>"
        "</div>"
    ),
)

with gr.Blocks(theme=THEME, title="ML for Trading — Teaching Assistant") as demo:
    gr.ChatInterface(
        fn=ask_question,
        chatbot=chatbot,
        title="🎓 ML for Trading — Teaching Assistant",
        description="Ask anything from the **Machine Learning for Trading** course by Professor Tucker Balch.",
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
