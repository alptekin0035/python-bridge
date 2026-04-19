from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import openai
import anthropic
import base64
import os
import uuid
import httpx
from typing import Optional

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")

class TTSRequest(BaseModel):
    text: str
    voice: str = "onyx"

class ImageEditRequest(BaseModel):
    image_url: str
    prompt: str

class ImageGenerateRequest(BaseModel):
    prompt: str

class ImageAnalyzeRequest(BaseModel):
    image_url: str
    prompt: str

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

@app.post("/image_edit")
async def image_edit(req: ImageEditRequest):
    try:
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        async with httpx.AsyncClient() as client:
            img_response = await client.get(req.image_url, headers=headers)
            img_bytes = img_response.content

        file_id = str(uuid.uuid4())
        input_path = f"/tmp/{file_id}_input.png"
        output_path = f"/tmp/{file_id}_output.png"

        with open(input_path, "wb") as f:
            f.write(img_bytes)

        oai = openai.OpenAI(api_key=OPENAI_API_KEY)
        with open(input_path, "rb") as img_file:
            response = oai.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=req.prompt,
                n=1,
                size="1024x1024"
            )

        edited_b64 = response.data[0].b64_json
        edited_bytes = base64.b64decode(edited_b64)
        with open(output_path, "wb") as f:
            f.write(edited_bytes)

        return {"file_id": file_id, "url_path": f"/image/{file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image_generate")
async def image_generate(req: ImageGenerateRequest):
    try:
        oai = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = oai.images.generate(
            model="gpt-image-1",
            prompt=req.prompt,
            n=1,
            size="1024x1024"
        )
        edited_b64 = response.data[0].b64_json
        edited_bytes = base64.b64decode(edited_b64)

        file_id = str(uuid.uuid4())
        output_path = f"/tmp/{file_id}_output.png"
        with open(output_path, "wb") as f:
            f.write(edited_bytes)

        return {"file_id": file_id, "url_path": f"/image/{file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/{file_id}")
async def get_image(file_id: str):
    output_path = f"/tmp/{file_id}_output.png"
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(output_path, media_type="image/png")

@app.post("/analyze_image")
async def analyze_image(req: ImageAnalyzeRequest):
    try:
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        async with httpx.AsyncClient() as http_client:
            img_response = await http_client.get(req.image_url, headers=headers)
            img_bytes = img_response.content

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": req.prompt
                        }
                    ]
                }
            ]
        )
        return {"result": response.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
