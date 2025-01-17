from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import your router
from routers.summarize import router as summarize_router

app = FastAPI()

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the summarize router
app.include_router(summarize_router, prefix="")

@app.get("/api")
def root():
    return {"message": "Welcome to the YouTube Summarization API!"}