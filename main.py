from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import logging
import sys
import os
import uuid
import asyncio
import soundfile as sf
import numpy as np
from typing import Optional, List
import tempfile
import threading
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories for audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Global TTS model variable
tts_model = None
model_lock = threading.Lock()

# --- FIX 1: DYNAMIC BASE URL HANDLING ---
# Reads the public-facing URL from an environment variable set by the workflow.
# Falls back to a placeholder if not set, preventing crashes.
BASE_URL = os.getenv('BASE_URL', 'https://placeholder-url-not-set.trycloudflare.com')
if 'placeholder' in BASE_URL:
    logger.warning(f"BASE_URL environment variable not found. Using placeholder: {BASE_URL}")
else:
    logger.info(f"Public URL successfully loaded: {BASE_URL}")

# --- FIX 2: API METRICS TRACKING ---
# In-memory dictionary to store real-time metrics for the /metrics endpoint.
api_metrics = {
    "tts_requests_total": 0,
    "tts_requests_success": 0,
    "tts_requests_failed": 0,
}

# Create FastAPI app
app = FastAPI(
    title="KittenTTS API",
    description="Real KittenTTS implementation for n8n and Telegram integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving audio
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Request models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "expr-voice-2-f"
    speed: Optional[float] = 1.0
    language: Optional[str] = "en"
    format: Optional[str] = "wav"  # KittenTTS outputs wav by default
    telegram_chat_id: Optional[str] = None
    telegram_bot_token: Optional[str] = None

class TTSResponse(BaseModel):
    text: str
    status: str
    audio_url: Optional[str] = None
    audio_file_path: Optional[str] = None
    file_id: Optional[str] = None
    message: Optional[str] = None
    telegram_ready: Optional[bool] = False
    duration: Optional[float] = None
    sample_rate: Optional[int] = None

class VoiceInfo(BaseModel):
    name: str
    gender: str
    language: str
    description: str

def initialize_kitten_tts():
    """Initialize KittenTTS model"""
    global tts_model
    try:
        logger.info("Initializing KittenTTS model...")
        
        # Import KittenTTS
        from kittentts import KittenTTS
        
        # Initialize model with the nano version
        tts_model = KittenTTS("KittenML/kitten-tts-nano-0.1")
        logger.info("✅ KittenTTS model loaded successfully!")
        return True
        
    except ImportError as e:
        logger.error("❌ KittenTTS not found. Installing...")
        try:
            import subprocess
            import sys
            
            # Try to install KittenTTS
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl"
            ])
            
            # Try importing again
            from kittentts import KittenTTS
            tts_model = KittenTTS("KittenML/kitten-tts-nano-0.1")
            logger.info("✅ KittenTTS installed and loaded successfully!")
            return True
            
        except Exception as install_error:
            logger.error(f"❌ Failed to install KittenTTS: {install_error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize KittenTTS: {e}")
        return False

def get_available_voices() -> List[VoiceInfo]:
    """Get list of available KittenTTS voices"""
    voices = [
        VoiceInfo(name="expr-voice-2-m", gender="male", language="en", description="Expressive male voice 2"),
        VoiceInfo(name="expr-voice-2-f", gender="female", language="en", description="Expressive female voice 2"),
        VoiceInfo(name="expr-voice-3-m", gender="male", language="en", description="Expressive male voice 3"),
        VoiceInfo(name="expr-voice-3-f", gender="female", language="en", description="Expressive female voice 3"),
        VoiceInfo(name="expr-voice-4-m", gender="male", language="en", description="Expressive male voice 4"),
        VoiceInfo(name="expr-voice-4-f", gender="female", language="en", description="Expressive female voice 4"),
        VoiceInfo(name="expr-voice-5-m", gender="male", language="en", description="Expressive male voice 5"),
        VoiceInfo(name="expr-voice-5-f", gender="female", language="en", description="Expressive female voice 5")
    ]
    return voices

async def generate_audio_file(text: str, voice: str = "expr-voice-2-f", speed: float = 1.0, format: str = "wav") -> tuple:
    """
    Generate audio file from text using KittenTTS
    Returns: (filename, duration, sample_rate)
    """
    global tts_model
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not initialized")
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.{format.lower()}"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        logger.info(f"Generating audio: '{text[:50]}...' with voice '{voice}'")
        
        # Generate audio using KittenTTS
        with model_lock:
            # KittenTTS generate method
            audio_data = tts_model.generate(text, voice=voice)
        
        # Apply speed modification if needed
        if speed != 1.0:
            # Simple speed modification using numpy
            if speed > 1.0:
                # Speed up by skipping samples
                step = int(speed)
                audio_data = audio_data[::step]
            elif speed < 1.0:
                # Slow down by interpolating samples
                import scipy.signal
                audio_data = scipy.signal.resample(
                    audio_data, 
                    int(len(audio_data) / speed)
                )
        
        # KittenTTS outputs at 24kHz sample rate
        sample_rate = 24000
        duration = len(audio_data) / sample_rate
        
        # Save audio file
        sf.write(file_path, audio_data, sample_rate)
        
        logger.info(f"✅ Audio generated: {filename} ({duration:.2f}s)")
        
        # Convert to other formats if requested
        if format.lower() != "wav":
            await convert_audio_format(file_path, format.lower())
        
        return filename, duration, sample_rate
        
    except Exception as e:
        logger.error(f"❌ Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

async def convert_audio_format(wav_path: str, target_format: str):
    """Convert WAV to other formats"""
    try:
        if target_format == "mp3":
            # Convert WAV to MP3 using pydub if available
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(wav_path)
                mp3_path = wav_path.replace(".wav", ".mp3")
                audio.export(mp3_path, format="mp3")
                os.remove(wav_path)  # Remove original WAV
                logger.info(f"✅ Converted to MP3: {mp3_path}")
            except ImportError:
                logger.warning("pydub not available, keeping WAV format")
                
        elif target_format == "ogg":
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(wav_path)
                ogg_path = wav_path.replace(".wav", ".ogg")
                audio.export(ogg_path, format="ogg")
                os.remove(wav_path)  # Remove original WAV
                logger.info(f"✅ Converted to OGG: {ogg_path}")
            except ImportError:
                logger.warning("pydub not available, keeping WAV format")
                
    except Exception as e:
        logger.warning(f"Format conversion failed: {e}, keeping original format")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 KittenTTS API is starting up...")
    
    # Initialize TTS model in background
    def init_model():
        success = initialize_kitten_tts()
        if success:
            logger.info("🎵 TTS engine ready!")
        else:
            logger.error("❌ TTS engine failed to initialize - running in demo mode")
    
    # Run model initialization in a separate thread
    threading.Thread(target=init_model, daemon=True).start()

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    model_status = "✅ Ready" if tts_model is not None else "⏳ Loading..."
    
    return {
        "message": "🐱 KittenTTS FastAPI is running!",
        "version": "1.0.0",
        "status": "active",
        "tts_model_status": model_status,
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics", # Added for clarity
            "tts": "/tts (POST)",
            "voices": "/voices (GET)",
            "audio": "/audio/{filename} (GET)",
            "docs": "/docs"
        },
        "supported_voices": len(get_available_voices()),
        "features": ["Real KittenTTS", "Multiple voices", "Speed control", "Telegram ready"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_ready = tts_model is not None
    
    return {
        "status": "healthy" if model_ready else "loading",
        "service": "KittenTTS API",
        "model_loaded": model_ready,
        "timestamp": datetime.now().isoformat(),
        "audio_dir": AUDIO_DIR,
        "audio_files_count": len(os.listdir(AUDIO_DIR)) if os.path.exists(AUDIO_DIR) else 0
    }

@app.get("/metrics")
async def get_metrics():
    """[NEW] Metrics endpoint for comprehensive monitoring."""
    try:
        audio_files_count = len(os.listdir(AUDIO_DIR))
    except Exception as e:
        logger.error(f"Could not read audio directory for metrics: {e}")
        audio_files_count = -1  # Indicate an error

    # Create a copy to ensure thread safety
    current_metrics = api_metrics.copy()
    current_metrics.update({
        "model_loaded": tts_model is not None,
        "current_audio_files": audio_files_count,
        "timestamp": datetime.now().isoformat()
    })
    return JSONResponse(content=current_metrics)

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Convert text to speech using real KittenTTS
    Perfect for n8n workflows and Telegram integration
    """
    api_metrics["tts_requests_total"] += 1
    try:
        logger.info(f"📝 TTS request: '{request.text[:50]}...' voice='{request.voice}'")
        
        # Validation
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        # Check if model is loaded
        if tts_model is None:
            raise HTTPException(status_code=503, detail="TTS model is still loading, please wait...")
        
        # Validate voice
        available_voices = [v.name for v in get_available_voices()]
        if request.voice not in available_voices:
            logger.warning(f"Unknown voice '{request.voice}', using default")
            request.voice = "expr-voice-2-f"
        
        # Generate audio file
        filename, duration, sample_rate = await generate_audio_file(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            format=request.format
        )
        
        # --- FIX 1 in action: Use the dynamic BASE_URL ---
        audio_url = f"{BASE_URL}/audio/{filename}"
        
        response = TTSResponse(
            text=request.text,
            status="success",
            audio_url=audio_url,
            audio_file_path=f"/audio/{filename}",
            file_id=filename.split('.')[0],
            message=f"🎵 Audio generated with voice '{request.voice}' (speed: {request.speed}x, duration: {duration:.2f}s)",
            telegram_ready=True,
            duration=duration,
            sample_rate=sample_rate
        )
        
        logger.info(f"✅ TTS completed: {filename}")
        
        api_metrics["tts_requests_success"] += 1
        # Schedule cleanup after 1 hour
        background_tasks.add_task(cleanup_file, filename, 3600)
        
        return response
        
    except HTTPException as e:
        api_metrics["tts_requests_failed"] += 1
        raise # Re-raise the exception after counting it
    except Exception as e:
        api_metrics["tts_requests_failed"] += 1
        logger.error(f"❌ TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

async def cleanup_file(filename: str, delay: int):
    """Clean up audio file after delay"""
    await asyncio.sleep(delay)
    try:
        file_path = os.path.join(AUDIO_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🧹 Cleaned up: {filename}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {filename}: {e}")

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio files for download/streaming"""
    file_path = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Determine media type
    extension = filename.split('.')[-1].lower()
    media_types = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'ogg': 'audio/ogg'
    }
    
    media_type = media_types.get(extension, 'application/octet-stream')
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
        headers={
            "Content-Disposition": f"inline; filename={filename}",
            "Cache-Control": "public, max-age=3600"
        }
    )

@app.get("/voices")
async def get_voices():
    """Get list of available KittenTTS voices"""
    voices = get_available_voices()
    return {
        "voices": [voice.dict() for voice in voices],
        "total": len(voices),
        "default": "expr-voice-2-f",
        "note": "KittenTTS supports 8 high-quality voices"
    }

@app.get("/formats")
async def get_supported_formats():
    """Get supported audio formats"""
    return {
        "formats": [
            {"format": "wav", "mime_type": "audio/wav", "telegram_compatible": True, "default": True},
            {"format": "mp3", "mime_type": "audio/mpeg", "telegram_compatible": True, "requires": "pydub"},
            {"format": "ogg", "mime_type": "audio/ogg", "telegram_compatible": True, "requires": "pydub"}
        ],
        "recommended": "wav",
        "note": "WAV is native KittenTTS format, others require pydub"
    }

@app.get("/model-info")
async def get_model_info():
    """Get KittenTTS model information"""
    return {
        "model_name": "KittenML/kitten-tts-nano-0.1",
        "model_size": "~25MB",
        "parameters": "15M",
        "sample_rate": 24000,
        "cpu_optimized": True,
        "gpu_required": False,
        "languages": ["en"],
        "voices_count": 8,
        "real_time_capable": True,
        "loaded": tts_model is not None
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service": "🐱 KittenTTS API",
        "status": "running",
        "model_loaded": tts_model is not None,
        "python_version": sys.version.split()[0],
        "fastapi_version": "0.100+",
        "audio_directory": AUDIO_DIR,
        "total_audio_files": len(os.listdir(AUDIO_DIR)) if os.path.exists(AUDIO_DIR) else 0,
        "features": {
            "real_kitten_tts": True,
            "multiple_voices": True,
            "speed_control": True,
            "multiple_formats": True,
            "telegram_ready": True,
            "n8n_compatible": True,
            "cpu_optimized": True,
            "lightweight": "25MB model"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/test-voice")
async def test_voice(voice: str = "expr-voice-2-f", background_tasks: BackgroundTasks = BackgroundTasks()):
    """Test a specific voice with sample text"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    test_text = f"Hello! This is a test of the {voice} voice from KittenTTS."
    
    request = TTSRequest(
        text=test_text,
        voice=voice,
        speed=1.0,
        format="wav"
    )
    
    return await text_to_speech(request, background_tasks)

@app.delete("/cleanup")
async def cleanup_old_files():
    """Clean up old audio files"""
    try:
        if not os.path.exists(AUDIO_DIR):
            return {"message": "No audio directory found"}
        
        files = os.listdir(AUDIO_DIR)
        deleted_count = 0
        
        for file in files:
            file_path = os.path.join(AUDIO_DIR, file)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete {file}: {str(e)}")
        
        return {
            "message": f"🧹 Cleanup completed",
            "deleted_files": deleted_count,
            "remaining_files": len(os.listdir(AUDIO_DIR))
        }
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
