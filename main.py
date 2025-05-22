from fastapi import FastAPI
from pydantic import BaseModel
from aa import ContentBasedRecommender

app = FastAPI()
recommender = ContentBasedRecommender("duygu_400.xlsx")

class InputModel(BaseModel):
    duygu1: str
    duygu2: str
    duygu3: str
    sosyallik: str
    zaman: str
    top_k: int = 3

@app.post("/oner")
def get_recommendation(data: InputModel):
    result = recommender.recommend(
        duygu1=data.duygu1,
        duygu2=data.duygu2,
        duygu3=data.duygu3,
        sosyallik=data.sosyallik,
        zaman=data.zaman,
        top_k=data.top_k
    )
    return result.to_dict(orient="records")
