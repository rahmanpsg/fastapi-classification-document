from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FastAPI",
              description="Aplikasi Klasifikasi dan Pencarian Menggunakan Metode K-NN")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost",
                   "http://localhost:8080", os.getenv('CLIENT_URL')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
def read_root():
    return {"Hello": "World"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
