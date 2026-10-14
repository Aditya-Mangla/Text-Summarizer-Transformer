import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Folder where the notebook saved the fine-tuned model + tokenizer
MODEL_DIR = os.getenv("MODEL_DIR", "./my_saved_transformer")

ml = {}  # holds the model, tokenizer and device once loaded


def get_device() -> torch.device:
    # Same device logic as the notebook
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def process_dialogue(text: str) -> str:
    # Must match the preprocessing used during training
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    device = get_device()
    ml["tokenizer"] = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    ml["model"] = model
    ml["device"] = device
    yield
    ml.clear()


app = FastAPI(title="Text Summarizer API", lifespan=lifespan)

# Lets the frontend call the API even if index.html is opened from another origin.
# Tighten allow_origins before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(Path(__file__).parent / "index.html")


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Dialogue or text to summarize")
    max_length: int = Field(150, ge=10, le=150, description="Max summary tokens")
    min_length: int = Field(10, ge=1, le=100, description="Min summary tokens")
    num_beams: int = Field(4, ge=1, le=8, description="Beam search width")


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(ml["device"])}


# Plain `def` (not async) so FastAPI runs the heavy model call in a worker thread
@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if req.min_length > req.max_length:
        raise HTTPException(status_code=400, detail="min_length cannot exceed max_length")

    cleaned = process_dialogue(req.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning")

    tokenizer, model, device = ml["tokenizer"], ml["model"], ml["device"]

    inputs = tokenizer(
        cleaned, return_tensors="pt", max_length=512, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=req.max_length,
            min_length=req.min_length,
            num_beams=req.num_beams,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return SummarizeResponse(summary=summary)
