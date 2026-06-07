import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.CLI.query_rag import QueryResponse, query_rag
import gradio as gr
import yfinance as yf
from datetime import datetime

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

# ── Ticker bar ────────────────────────────────────────────────
TICKERS = {"^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "^NSEBANK": "BANK NIFTY"}

def fetch_ticker_html():
    spans = []
    for sym, label in TICKERS.items():
        try:
            t = yf.Ticker(sym)
            info = t.fast_info
            price = info.last_price
            prev  = info.previous_close
            chg_pct = ((price - prev) / prev) * 100 if prev else 0
            sign  = "+" if chg_pct >= 0 else ""
            color = "#88b878" if chg_pct >= 0 else "#c87060"
            spans.append(
                f'<span style="color:#dfc9a0;">'
                f'<span style="color:#f5e8cc;">{label}</span> '
                f'{price:,.2f} '
                f'<span style="color:{color};">{sign}{chg_pct:.1f}%</span>'
                f'</span>'
            )
        except Exception:
            spans.append(f'<span style="color:#dfc9a0;">{label} —</span>')

    updated = datetime.now().strftime("%H:%M:%S")
    return f"""
        <div style="
            background: #6b4e2a; padding: 7px 28px;
            border-bottom: 1px solid #c8b99a;
            display: flex; gap: 28px; align-items: center;
            font-family: Georgia, serif;
            font-size: 11px; letter-spacing: 0.06em;
        ">
            {''.join(spans)}
            <span style="margin-left:auto; color:#9a7448; font-size:10px;">
                updated {updated}
            </span>
        </div>
    """

# ── Gradio UI ─────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400&display=swap');
body { font-family: 'Georgia', serif; background: #ede8df; }
.gradio-container {
    max-width: 860px !important; margin: auto;
    background: #faf7f2 !important;
    border: 1px solid #c8b99a !important;
    border-radius: 2px !important;
    box-shadow: 0 1px 6px rgba(80,55,20,0.10) !important;
    padding: 0 !important; overflow: hidden;
}
#chatbot {
    border: none !important; border-radius: 0 !important;
    background: #faf7f2 !important;
    padding: 24px 28px !important; min-height: 340px;
    border-bottom: 1px solid #c8b99a !important;
}
#chatbot .wrap { background: #faf7f2 !important; }
.message.user {
    background: #7a5c38 !important; color: #fdf6e8 !important;
    border-radius: 2px !important; border: none !important;
    border-left: 3px solid #e8c87a !important;
    font-size: 14px !important; line-height: 1.7 !important;
    font-family: 'Georgia', serif !important;
}
.message.bot {
    background: #f0e8d8 !important; color: #3a2a18 !important;
    border-radius: 2px !important; border: none !important;
    border-left: 3px solid #b89060 !important;
    font-size: 14px !important; line-height: 1.7 !important;
    font-family: 'Georgia', serif !important;
}
.input-row {
    padding: 16px 28px !important; background: #ede8df !important;
    display: flex; gap: 10px; align-items: center;
}
#msg {
    border: 1px solid #c8b99a !important; border-radius: 2px !important;
    background: #faf7f2 !important; font-family: 'Georgia', serif !important;
    font-size: 14px !important; color: #3a2a18 !important;
    padding: 10px 14px !important;
}
#msg:focus {
    outline: none !important; border-color: #9a7448 !important;
    box-shadow: 0 0 0 2px rgba(122,92,56,0.12) !important;
}
#msg::placeholder { color: #b89878 !important; }
#send-btn {
    background: #7a5c38 !important; color: #fdf6e8 !important;
    border: none !important; border-radius: 2px !important;
    font-family: 'Georgia', serif !important; font-size: 13px !important;
    letter-spacing: 0.06em !important; padding: 10px 22px !important;
}
#send-btn:hover { background: #6b4e2a !important; }
#clear-btn {
    background: transparent !important; color: #9a7448 !important;
    border: 1px solid #c8b99a !important; border-radius: 2px !important;
    font-family: 'Georgia', serif !important; font-size: 13px !important;
    padding: 10px 14px !important;
}
#clear-btn:hover { border-color: #9a7448 !important; color: #7a5c38 !important; }
footer { display: none !important; }
"""

def respond(message, history):
    history = history or []
    try:
        response = query_rag(message)
        history.append((message, response.answer))
    except Exception:
        history.append((message, "⚠️ Model is temporarily busy. Please try again in a moment."))
    return "", history

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("Merriweather"), "Georgia", "serif"]
    ),
    css=custom_css,
    title="ML for Trading TA"
) as demo:

    gr.HTML("""
        <div style="
            background: #7a5c38; padding: 16px 28px;
            display: flex; align-items: center; gap: 14px;
        ">
            <div style="width:8px;height:8px;border-radius:50%;background:#e8c87a;"></div>
            <span style="font-family:Georgia,serif;font-size:15px;font-weight:normal;
                letter-spacing:0.12em;text-transform:uppercase;color:#fdf6e8;">
                ML for Trading TA
            </span>
            <div style="margin-left:auto;display:flex;gap:8px;">
                <span style="font-family:Georgia,serif;font-size:10px;color:#e2c99a;
                    padding:3px 10px;border:1px solid #a07848;border-radius:1px;
                    letter-spacing:0.12em;">SIGNALS</span>
                <span style="font-family:Georgia,serif;font-size:10px;color:#e2c99a;
                    padding:3px 10px;border:1px solid #a07848;border-radius:1px;
                    letter-spacing:0.12em;">MODELS</span>
                <span style="font-family:Georgia,serif;font-size:10px;color:#e2c99a;
                    padding:3px 10px;border:1px solid #a07848;border-radius:1px;
                    letter-spacing:0.12em;">HISTORY</span>
            </div>
        </div>
    """)

    ticker_display = gr.HTML(value=fetch_ticker_html)

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        render_markdown=True
    )

    with gr.Row(elem_classes=["input-row"]):
        msg = gr.Textbox(
            placeholder="Ask about RSI, MACD, reinforcement learning…",
            show_label=False, container=False,
            elem_id="msg", scale=8
        )
        send  = gr.Button("Send",  elem_id="send-btn",  scale=1, min_width=80)
        clear = gr.Button("Clear", elem_id="clear-btn", scale=1, min_width=60)

    timer = gr.Timer(60)
    timer.tick(fn=fetch_ticker_html, outputs=ticker_display)

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond,  [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

# ── Mount Gradio on FastAPI ───────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)