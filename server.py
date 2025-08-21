from fastapi import FastAPI
from fastapi.responses import FileResponse
import soundfile as sf
from kittentts import KittenTTS

app = FastAPI()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/tts")
async def tts(text: str):
    m = KittenTTS("KittenML/kitten-tts-nano-0.1")
    audio = m.generate(text)
    sf.write("output.wav", audio, 24000)
    return FileResponse("output.wav", media_type="audio/wav", filename="output.wav")
