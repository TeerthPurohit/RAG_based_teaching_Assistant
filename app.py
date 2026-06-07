import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.CLI.query_rag import QueryResponse, query_rag
import gradio as gr

app = FastAPI()

# ── FastAPI routes ────────────────────────────────────────────
class SubmitQueryRequest(BaseModel):
    query_text: str

@app.get("/")
def index():
    return {"message": "RAG Teaching Assistant API"}

@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryResponse:
    query_response = query_rag(request.query_text)
    return query_response

# ── Gradio UI ─────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
body { font-family: 'Inter', system-ui, sans-serif; background: #0d1117; }
.gradio-container {
    max-width: 860px !important; margin: auto;
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
    padding: 0 !important; overflow: hidden;
}
#chatbot {
    border: none !important; border-radius: 0 !important;
    background: #0d1117 !important; padding: 24px 28px !important;
    min-height: 340px; border-bottom: 1px solid #21262d !important;
}
#chatbot .wrap { background: #0d1117 !important; }
.message.user {
    background: #1f2d3d !important; color: #cae8ff !important;
    border-radius: 5px !important; border: 1px solid #1f6feb !important;
    border-left: 3px solid #1f6feb !important;
    font-size: 13px !important; line-height: 1.65 !important;
}
.message.bot {
    background: #161b22 !important; color: #c9d1d9 !important;
    border-radius: 5px !important; border: 1px solid #30363d !important;
    border-left: 3px solid #3fb950 !important;
    font-size: 13px !important; line-height: 1.65 !important;
}
.input-row {
    padding: 14px 28px !important; background: #161b22 !important;
    border-top: 1px solid #21262d !important;
    display: flex; gap: 10px; align-items: center;
}
#msg {
    border: 1px solid #30363d !important; border-radius: 4px !important;
    background: #0d1117 !important; font-size: 13px !important;
    color: #e6edf3 !important; padding: 9px 14px !important;
}
#msg:focus {
    outline: none !important; border-color: #388bfd !important;
    box-shadow: 0 0 0 2px rgba(56,139,253,0.15) !important;
}
#msg::placeholder { color: #484f58 !important; }
#send-btn {
    background: #238636 !important; color: #ffffff !important;
    border: 1px solid #2ea043 !important; border-radius: 4px !important;
    font-size: 12px !important; font-weight: 600 !important;
    padding: 9px 22px !important;
}
#send-btn:hover { background: #2ea043 !important; }
#clear-btn {
    background: transparent !important; color: #8b949e !important;
    border: 1px solid #30363d !important; border-radius: 4px !important;
    font-size: 12px !important; padding: 9px 14px !important;
}
#clear-btn:hover { border-color: #8b949e !important; color: #c9d1d9 !important; }
footer { display: none !important; }
"""

# ← back to tuples since HF's Gradio version doesn't support type="messages"
def respond(message, history):
    history = history or []
    try:
        response = query_rag(message)
        history.append((message, response.answer))
    except Exception as e:
        history.append((message, "⚠️ Model is temporarily busy. Please try again in a moment."))
    return "", history

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ),
    css=custom_css,
    title="ML for Trading TA"
) as demo:
    gr.HTML("""
        <div style="
            display: flex; align-items: center; gap: 10px;
            padding: 14px 28px;
            border-bottom: 1px solid #21262d;
            background: #161b22;
        ">
            <div style="width:8px; height:8px; background:#3fb950; border-radius:1px;"></div>
            <span style="
                font-family: Inter, system-ui, sans-serif;
                font-size: 13px; font-weight: 600;
                letter-spacing: 0.14em; text-transform: uppercase;
                color: #e6edf3;
            ">ML for Trading TA</span>
        </div>
    """)
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        render_markdown=True    # ← type="messages" removed
    )
    with gr.Row(elem_classes=["input-row"]):
        msg = gr.Textbox(
            placeholder="Ask about RSI, MACD, reinforcement learning…",
            show_label=False, container=False,
            elem_id="msg", scale=8
        )
        send = gr.Button("SEND", elem_id="send-btn", scale=1, min_width=80)
        clear = gr.Button("Clear", elem_id="clear-btn", scale=1, min_width=60)

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

# ── Mount Gradio on FastAPI ───────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)