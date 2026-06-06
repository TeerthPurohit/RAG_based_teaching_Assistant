import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.CLI.query_rag import QueryResponse, query_rag

app = FastAPI()

class SubmitQueryRequest(BaseModel):
    query_text: str

@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryResponse:
    query_response = query_rag(request.query_text)
    return query_response

if __name__ == "__main__":
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=port)