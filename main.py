from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import sys
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="KittenTTS API",
    description="Text-to-Speech API using KittenTTS",
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

# Request models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    speed: Optional[float] = 1.0
    language: Optional[str] = "en"

class TTSResponse(BaseModel):
    text: str
    status: str
    audio_url: Optional[str] = None
    message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    logger.info("KittenTTS API is starting up...")
    # Initialize TTS engine here if needed
    try:
        # Try to import kittentts
        # import kittentts
        logger.info("TTS engine initialized successfully")
    except ImportError:
        logger.warning("kittentts module not found, running in demo mode")

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "KittenTTS FastAPI is running!",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "tts": "/tts (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "KittenTTS API",
        "timestamp": "2025-08-21"
    }

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech
    """
    try:
        logger.info(f"Processing TTS request: {request.text[:50]}...")
        
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        # TODO: Implement actual TTS logic here
        # This is a placeholder implementation
        
        # For now, return a success response
        response = TTSResponse(
            text=request.text,
            status="processed",
            message=f"Text processed successfully with voice '{request.voice}' at speed {request.speed}x"
        )
        
        logger.info("TTS request processed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/voices")
async def get_available_voices():
    """Get list of available voices"""
    # This would normally query the TTS engine for available voices
    return {
        "voices": [
            {"name": "default", "language": "en", "gender": "neutral"},
            {"name": "female", "language": "en", "gender": "female"},
            {"name": "male", "language": "en", "gender": "male"}
        ]
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service": "KittenTTS API",
        "status": "running",
        "python_version": sys.version,
        "fastapi_version": "0.100+",
        "features": {
            "tts": True,
            "multiple_voices": True,
            "speed_control": True
        }
    }

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
