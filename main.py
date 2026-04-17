from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import openai
import base64
import os
import uuid

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
        file_id = str(uuid.uuid4())
        file_path = f"/tmp/{file_id}.mp3"
        with open(file_path, "wb") as f:
            f.write(response.content)
        return {"file_id": file_id, "url_path": f"/audio/{file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    file_path = f"/tmp/{file_id}.mp3"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(file_path, media_type="audio/mpeg")
