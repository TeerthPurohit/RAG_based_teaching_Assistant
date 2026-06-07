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
body { font-family: 'Inter', system-ui, sans-serif; background: #f0f2f5; }
.gradio-container {
    max-width: 860px !important; margin: auto;
    background: #ffffff !important;
    border: 1px solid #dde2ea !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 4px rgba(30,50,80,0.07) !important;
    padding: 0 !important; overflow: hidden;
}
#chatbot {
    border: none !important; border-radius: 0 !important;
    background: #ffffff !important; padding: 24px 28px !important;
    min-height: 340px; border-bottom: 1px solid #dde2ea !important;
}
.message.user {
    background: #1a56db !important; color: #ffffff !important;
    border-radius: 5px !important; border: 1px solid #1a56db !important;
    font-size: 13px !important; line-height: 1.65 !important;
}
.message.bot {
    background: #f8fafc !important; color: #0f172a !important;
    border-radius: 5px !important; border: 1px solid #dde2ea !important;
    font-size: 13px !important; line-height: 1.65 !important;
}
.input-row {
    padding: 14px 28px !important; background: #f8fafc !important;
    border-top: 1px solid #dde2ea !important;
    display: flex; gap: 10px; align-items: center;
}
#msg {
    border: 1px solid #dde2ea !important; border-radius: 4px !important;
    background: #ffffff !important; font-size: 13px !important;
    color: #0f172a !important; padding: 9px 14px !important;
}
#msg:focus {
    outline: none !important; border-color: #1a56db !important;
    box-shadow: 0 0 0 2px rgba(26,86,219,0.12) !important;
}
#send-btn {
    background: #1a56db !important; color: #ffffff !important;
    border: none !important; border-radius: 4px !important;
    font-size: 12px !important; font-weight: 600 !important;
    padding: 9px 22px !important; cursor: pointer !important;
}
#send-btn:hover { background: #1648c0 !important; }
#clear-btn {
    background: transparent !important; color: #64748b !important;
    border: 1px solid #dde2ea !important; border-radius: 4px !important;
    font-size: 12px !important; padding: 9px 14px !important;
}
footer { display: none !important; }
"""

def respond(message, history):
    history = history or []
    response = query_rag(message)          # ← calls your real model
    history.append((message, response.answer))
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
            padding: 16px 28px;
            border-bottom: 2px solid #1a56db;
            background: #ffffff;
        ">
            <div style="width:8px; height:8px; background:#1a56db; border-radius:1px;"></div>
            <span style="
                font-family: Inter, system-ui, sans-serif;
                font-size: 13px; font-weight: 600;
                letter-spacing: 0.14em; text-transform: uppercase;
                color: #0f172a;
            ">ML for Trading TA</span>
        </div>
    """)
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        render_markdown=True         # ← markdown rendering enabled
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