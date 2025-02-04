from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Definição do modelo de entrada usando Pydantic
class TextData(BaseModel):
    text: str

# Criação do app FastAPI
app = FastAPI(title="API de Análise de Sentimentos")

# Carrega o pipeline de análise de sentimentos
# Esse pipeline já vem pré-treinado para tarefas de sentiment-analysis
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

@app.post("/analisar")
def analisar_texto(data: TextData):
    try:
        result = sentiment_pipeline(data.text)
        return {"resultado": result}
    except Exception as e:
        return {"erro": str(e)}
