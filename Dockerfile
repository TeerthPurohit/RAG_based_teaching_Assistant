FROM python:3.12-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]