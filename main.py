from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import base64
import os

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class TTSRequest(BaseModel):
    text: str
    voice: str = "onyx"

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.audio.speech.create(
            model="tts-1",
            voice=req.voice,
            input=req.text
        )
        audio_base64 = base64.b64encode(response.content).decode("utf-8")
        return {"audio_base64": audio_base64, "format": "mp3"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))