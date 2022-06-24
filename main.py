
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from starlette.staticfiles import StaticFiles
from services.knn import KNN
from schemas.knn_schema import KNNBase

load_dotenv()


app = FastAPI(title="FastAPI",
              description="Aplikasi Klasifikasi dan Pencarian Menggunakan Metode K-NN")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost",
                   "http://localhost:8080",
                   "*", os.getenv('CLIENT_URL')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


KNN = KNN()


@app.post("/api/proses")
def prosesKNN(knn: KNNBase):
    print(knn.k)
    return KNN.proses(knn.text, knn.k)


app.mount("/", StaticFiles(directory="static", html=True), name="site")
