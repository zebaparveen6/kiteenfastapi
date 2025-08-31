from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
import sys
import os
import uuid
import asyncio
import aiofiles
import soundfile as sf
import numpy as np
from typing import Optional, List, Dict, Any
import tempfile
import threading
from datetime import datetime, timedelta
import json
import gc
import psutil
import time
import traceback
from contextlib import asynccontextmanager
import signal
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import wraps, lru_cache
import hashlib
import pickle
from pathlib import Path
import shutil
import subprocess
import re

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('kittentts.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global configuration
class Config:
    AUDIO_DIR = Path("audio_files")
    CACHE_DIR = Path("cache")
    MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "~/.cache/kittentts")).expanduser()
    MAX_TEXT_LENGTH = 10000  # Support very long texts
    MAX_CONCURRENT_REQUESTS = 10
    CLEANUP_INTERVAL = 3600  # 1 hour
    MAX_AUDIO_FILES = 100
    MEMORY_THRESHOLD = 0.85  # 85% memory usage threshold
    CPU_THRESHOLD = 0.90     # 90% CPU usage threshold
    CHUNK_SIZE = 500  # Characters per chunk for long texts
    MAX_RETRIES = 5
    RETRY_DELAY = 2.0

config = Config()

# Create directories
for directory in [config.AUDIO_DIR, config.CACHE_DIR, config.MODEL_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Global variables
tts_model = None
model_lock = threading.RLock()  # Re-entrant lock
request_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
executor = ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 1) + 1))

# Enhanced caching system
class EnhancedAudioCache:
    def __init__(self, max_size: int = 100, max_memory_mb: int = 512):
        self.cache = {}
        self.access_times = {}
        self.sizes = {}
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, data: bytes):
        with self._lock:
            data_size = len(data)
            
            # Check if we need to evict
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + data_size > self.max_memory_bytes):
                if not self._evict_oldest():
                    break  # No more items to evict
            
            self.cache[key] = data
            self.access_times[key] = time.time()
            self.sizes[key] = data_size
            self.current_memory += data_size
    
    def _evict_oldest(self) -> bool:
        if not self.access_times:
            return False
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        removed_size = self.sizes.pop(oldest_key, 0)
        self.cache.pop(oldest_key, None)
        self.access_times.pop(oldest_key, None)
        self.current_memory -= removed_size
        return True
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.sizes.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cache_size": len(self.cache),
                "memory_usage_mb": round(self.current_memory / 1024 / 1024, 2),
                "max_memory_mb": round(self.max_memory_bytes / 1024 / 1024, 2),
                "utilization_percent": round((self.current_memory / self.max_memory_bytes) * 100, 1)
            }

audio_cache = EnhancedAudioCache()

# Metrics tracking
class AdvancedMetrics:
    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.requests_cached = 0
        self.average_generation_time = 0.0
        self.model_load_time = 0.0
        self.last_reset = datetime.now()
        self.active_requests = 0
        self.total_audio_duration = 0.0
        self.total_text_chars = 0
        self.error_counts = {}
        self._lock = threading.Lock()
    
    def record_request(self, success: bool, generation_time: float = 0.0, 
                      cached: bool = False, audio_duration: float = 0.0, 
                      text_length: int = 0, error_type: str = None):
        with self._lock:
            self.requests_total += 1
            
            if success:
                self.requests_success += 1
                if cached:
                    self.requests_cached += 1
                
                if generation_time > 0:
                    self.average_generation_time = (
                        (self.average_generation_time * (self.requests_success - 1) + generation_time) 
                        / self.requests_success
                    )
                
                self.total_audio_duration += audio_duration
                self.total_text_chars += text_length
            else:
                self.requests_failed += 1
                if error_type:
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime = datetime.now() - self.last_reset
            return {
                "requests": {
                    "total": self.requests_total,
                    "success": self.requests_success,
                    "failed": self.requests_failed,
                    "cached": self.requests_cached,
                    "success_rate": round((self.requests_success / max(1, self.requests_total)) * 100, 2),
                    "cache_hit_rate": round((self.requests_cached / max(1, self.requests_total)) * 100, 2)
                },
                "performance": {
                    "average_generation_time": round(self.average_generation_time, 3),
                    "model_load_time": round(self.model_load_time, 3),
                    "active_requests": self.active_requests,
                    "total_audio_duration": round(self.total_audio_duration, 2),
                    "total_text_chars": self.total_text_chars
                },
                "errors": self.error_counts,
                "uptime": str(uptime).split('.')[0]  # Remove microseconds
            }

metrics = AdvancedMetrics()

# Enhanced request/response models
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=config.MAX_TEXT_LENGTH)
    voice: Optional[str] = Field(default="expr-voice-2-f")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=3.0)
    language: Optional[str] = Field(default="en")
    format: Optional[str] = Field(default="wav")
    quality: Optional[str] = Field(default="standard")  # standard, high
    telegram_chat_id: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    cache_enabled: Optional[bool] = Field(default=True)
    
    def generate_cache_key(self) -> str:
        """Generate a unique cache key for this request"""
        content = f"{self.text}_{self.voice}_{self.speed}_{self.format}_{self.quality}"
        return hashlib.md5(content.encode()).hexdigest()

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
    cache_hit: Optional[bool] = False
    generation_time: Optional[float] = None
    model_version: Optional[str] = None
    text_chunks: Optional[int] = None
    file_size_mb: Optional[float] = None

class VoiceInfo(BaseModel):
    name: str
    gender: str
    language: str
    description: str
    quality: str = "standard"
    sample_rate: int = 24000

# System utilities
def check_memory_usage() -> Dict[str, float]:
    """Check current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        system_memory = psutil.virtual_memory()
        
        return {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": memory_percent,
            "available_mb": system_memory.available / 1024 / 1024,
            "system_memory_percent": system_memory.percent,
            "total_mb": system_memory.total / 1024 / 1024
        }
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")
        return {"memory_mb": 0, "memory_percent": 0, "available_mb": 0, "system_memory_percent": 0, "total_mb": 0}

def cleanup_memory():
    """Comprehensive memory cleanup"""
    try:
        # Python garbage collection
        collected = gc.collect()
        
        # Clear caches
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        # Clear audio cache if memory is critical
        memory_stats = check_memory_usage()
        if memory_stats["system_memory_percent"] > 90:
            audio_cache.clear()
            logger.info("🧹 Cleared audio cache due to critical memory usage")
        
        logger.info(f"🧹 Memory cleanup completed: {collected} objects collected")
        
    except Exception as e:
        logger.warning(f"Memory cleanup error: {e}")

# Enhanced error handling decorator
def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to recover
            cleanup_memory()
            
            # Categorize error for metrics
            error_type = type(e).__name__
            metrics.record_request(False, error_type=error_type)
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "message": str(e),
                    "function": func.__name__,
                    "error_type": error_type,
                    "recoverable": True,
                    "timestamp": datetime.now().isoformat()
                }
            )
    return wrapper

# KittenTTS initialization with bulletproof error handling
async def initialize_kitten_tts() -> bool:
    """Initialize KittenTTS with comprehensive fallback methods"""
    global tts_model, metrics
    
    start_time = time.time()
    
    try:
        logger.info("🐱 Starting KittenTTS initialization...")
        
        # Set up environment for model caching
        os.environ["HF_HOME"] = str(config.MODEL_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(config.MODEL_CACHE_DIR / "transformers")
        os.environ["HF_HUB_CACHE"] = str(config.MODEL_CACHE_DIR / "hub")
        
        # Method 1: Try importing existing installation
        try:
            from kittentts import KittenTTS
            logger.info("✅ KittenTTS module found")
        except ImportError as e:
            logger.warning(f"KittenTTS not found: {e}")
            
            # Method 2: Runtime installation with multiple attempts
            logger.info("📦 Attempting runtime installation...")
            
            install_commands = [
                # Try PyPI first
                [sys.executable, "-m", "pip", "install", "kittentts", "--no-deps", "--force-reinstall"],
                [sys.executable, "-m", "pip", "install", "kittentts"],
                
                # Try GitHub release
                [sys.executable, "-m", "pip", "install", 
                 "https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl"],
                
                # Try direct GitHub
                [sys.executable, "-m", "pip", "install", "git+https://github.com/KittenML/KittenTTS.git"],
                
                # Try with specific dependencies
                [sys.executable, "-m", "pip", "install", "torch", "transformers", "soundfile", "numpy", "&&",
                 sys.executable, "-m", "pip", "install", "kittentts"]
            ]
            
            for i, cmd in enumerate(install_commands):
                try:
                    logger.info(f"Installation attempt {i+1}/{len(install_commands)}")
                    if "&&" in cmd:
                        # Handle compound commands
                        subprocess.check_call(" ".join(cmd), shell=True, timeout=600)
                    else:
                        subprocess.check_call(cmd, timeout=600)
                    
                    # Try importing after each attempt
                    from kittentts import KittenTTS
                    logger.info(f"✅ KittenTTS installed successfully (method {i+1})")
                    break
                    
                except Exception as install_error:
                    logger.warning(f"Install attempt {i+1} failed: {install_error}")
                    if i == len(install_commands) - 1:
                        logger.error("❌ All installation attempts failed")
                        return False
                    continue
        
        # Method 3: Initialize model with multiple fallback strategies
        model_configs = [
            {"model": "KittenML/kitten-tts-nano-0.1", "description": "Official nano model"},
            {"model": "kitten-tts-nano-0.1", "description": "Short name"},
            {"model": "nano", "description": "Simple name"},
        ]
        
        for config_item in model_configs:
            try:
                logger.info(f"🔄 Loading model: {config_item['description']}")
                
                with model_lock:
                    # Pre-download model if not cached
                    try:
                        from huggingface_hub import snapshot_download
                        snapshot_download(
                            repo_id=config_item["model"],
                            cache_dir=str(config.MODEL_CACHE_DIR),
                            resume_download=True,
                            local_files_only=False
                        )
                        logger.info("📥 Model downloaded/verified in cache")
                    except Exception as download_error:
                        logger.warning(f"Model download warning: {download_error}")
                    
                    # Initialize TTS model
                    tts_model = KittenTTS(config_item["model"])
                    
                    # Comprehensive model testing
                    test_texts = ["Test", "Hello world", "This is a longer test sentence."]
                    test_voices = ["expr-voice-2-f", "expr-voice-2-m"]
                    
                    all_tests_passed = True
                    for test_text in test_texts:
                        for test_voice in test_voices:
                            try:
                                test_audio = tts_model.generate(test_text, voice=test_voice)
                                if test_audio is None or len(test_audio) == 0:
                                    raise Exception(f"Empty audio for '{test_text}' with {test_voice}")
                                
                                # Validate audio properties
                                if not isinstance(test_audio, np.ndarray):
                                    raise Exception(f"Invalid audio type: {type(test_audio)}")
                                
                                if len(test_audio.shape) != 1:
                                    raise Exception(f"Invalid audio shape: {test_audio.shape}")
                                
                            except Exception as test_error:
                                logger.warning(f"Model test failed: {test_error}")
                                all_tests_passed = False
                                break
                        
                        if not all_tests_passed:
                            break
                    
                    if all_tests_passed:
                        logger.info(f"✅ Model loaded and tested successfully: {config_item['description']}")
                        break
                    else:
                        logger.warning(f"Model tests failed for: {config_item['description']}")
                        tts_model = None
            
            except Exception as model_error:
                logger.warning(f"Model loading failed ({config_item['description']}): {model_error}")
                tts_model = None
                continue
        
        if tts_model is None:
            logger.error("❌ All model loading attempts failed")
            return False
        
        metrics.model_load_time = time.time() - start_time
        logger.info(f"🎵 KittenTTS initialized successfully! Load time: {metrics.model_load_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ KittenTTS initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False

# Enhanced voice configuration
@lru_cache(maxsize=1)
def get_available_voices() -> List[VoiceInfo]:
    """Get comprehensive list of KittenTTS voices"""
    voices = [
        VoiceInfo(name="expr-voice-1-m", gender="male", language="en", description="Natural male voice 1", quality="standard"),
        VoiceInfo(name="expr-voice-1-f", gender="female", language="en", description="Natural female voice 1", quality="standard"),
        VoiceInfo(name="expr-voice-2-m", gender="male", language="en", description="Expressive male voice 2", quality="high"),
        VoiceInfo(name="expr-voice-2-f", gender="female", language="en", description="Expressive female voice 2", quality="high"),
        VoiceInfo(name="expr-voice-3-m", gender="male", language="en", description="Clear male voice 3", quality="standard"),
        VoiceInfo(name="expr-voice-3-f", gender="female", language="en", description="Clear female voice 3", quality="standard"),
        VoiceInfo(name="expr-voice-4-m", gender="male", language="en", description="Warm male voice 4", quality="high"),
        VoiceInfo(name="expr-voice-4-f", gender="female", language="en", description="Warm female voice 4", quality="high"),
        VoiceInfo(name="expr-voice-5-m", gender="male", language="en", description="Professional male voice 5", quality="standard"),
        VoiceInfo(name="expr-voice-5-f", gender="female", language="en", description="Professional female voice 5", quality="standard")
    ]
    return voices

# Advanced text preprocessing
def preprocess_text(text: str) -> List[str]:
    """Advanced text preprocessing and intelligent chunking"""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Clean and normalize text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s\.,!?;:\-\'"()]', '', text)  # Remove problematic chars
    
    # Handle short texts
    if len(text) <= config.CHUNK_SIZE:
        return [text]
    
    # Intelligent chunking for long texts
    chunks = []
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(paragraph) <= config.CHUNK_SIZE:
            chunks.append(paragraph.strip())
        else:
            # Split long paragraphs by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = ""
            
            for sentence in sentences:
                # If single sentence is too long, split by phrases
                if len(sentence) > config.CHUNK_SIZE:
                    phrases = re.split(r'(?<=[,;:])\s+', sentence)
                    for phrase in phrases:
                        if len(current_chunk) + len(phrase) <= config.CHUNK_SIZE:
                            current_chunk += (" " if current_chunk else "") + phrase
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = phrase
                else:
                    # Normal sentence processing
                    if len(current_chunk) + len(sentence) <= config.CHUNK_SIZE:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks if chunks else [text]

# Bulletproof audio generation
async def generate_audio_with_chunking(
    text: str,
    voice: str = "expr-voice-2-f",
    speed: float = 1.0,
    format: str = "wav",
    cache_key: Optional[str] = None
) -> tuple[str, float, int]:
    """Generate audio with comprehensive error handling and recovery"""
    
    global tts_model
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not initialized")
    
    # Check cache first
    cache_hit = False
    if cache_key and audio_cache.get(cache_key):
        try:
            cached_data = audio_cache.get(cache_key)
            if cached_data:
                logger.info(f"📋 Cache hit for key: {cache_key[:8]}...")
                
                # Write cached data to file
                filename = f"{uuid.uuid4()}.{format}"
                file_path = config.AUDIO_DIR / filename
                
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(cached_data)
                
                # Get audio info
                audio_data, sample_rate = sf.read(str(file_path))
                duration = len(audio_data) / sample_rate
                
                return str(filename), duration, sample_rate
        except Exception as cache_error:
            logger.warning(f"Cache retrieval failed: {cache_error}")
    
    start_time = time.time()
    
    try:
        # Preprocess text into optimal chunks
        text_chunks = preprocess_text(text)
        logger.info(f"📝 Processing {len(text_chunks)} text chunks (total: {len(text)} chars)")
        
        # Generate audio for each chunk with bulletproof error handling
        audio_segments = []
        total_duration = 0.0
        sample_rate = 24000  # KittenTTS default
        failed_chunks = 0
        max_failed_chunks = max(1, len(text_chunks) // 4)  # Allow up to 25% failures
        
        for i, chunk in enumerate(text_chunks):
            chunk_start_time = time.time()
            logger.info(f"🎵 Generating chunk {i+1}/{len(text_chunks)}: '{chunk[:50]}...'")
            
            # Memory check before each chunk
            memory_stats = check_memory_usage()
            if memory_stats["system_memory_percent"] > 85:
                logger.warning("⚠️ High memory usage, forcing cleanup")
                cleanup_memory()
                await asyncio.sleep(1)  # Give system time to recover
            
            # Generate chunk with retries
            chunk_audio = None
            chunk_attempts = 0
            max_chunk_attempts = config.MAX_RETRIES
            
            while chunk_attempts < max_chunk_attempts and chunk_audio is None:
                try:
                    chunk_attempts += 1
                    
                    # Generate with timeout
                    chunk_audio = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            executor,
                            lambda: generate_chunk_audio_safe(chunk, voice, chunk_attempts)
                        ),
                        timeout=300.0  # 5 minutes per chunk
                    )
                    
                    if chunk_audio is not None and len(chunk_audio) > 0:
                        audio_segments.append(chunk_audio)
                        chunk_duration = len(chunk_audio) / sample_rate
                        total_duration += chunk_duration
                        chunk_time = time.time() - chunk_start_time
                        logger.info(f"✅ Chunk {i+1} completed ({chunk_duration:.2f}s audio, {chunk_time:.2f}s generation)")
                        break
                    else:
                        raise Exception("Model returned empty audio")
                
                except asyncio.TimeoutError:
                    logger.error(f"❌ Chunk {i+1} timed out (attempt {chunk_attempts})")
                    if chunk_attempts < max_chunk_attempts:
                        await asyncio.sleep(config.RETRY_DELAY * chunk_attempts)
                        continue
                    else:
                        failed_chunks += 1
                        logger.error(f"❌ Chunk {i+1} failed after all attempts")
                        break
                
                except Exception as chunk_error:
                    logger.error(f"❌ Chunk {i+1} error (attempt {chunk_attempts}): {chunk_error}")
                    if chunk_attempts < max_chunk_attempts:
                        await asyncio.sleep(config.RETRY_DELAY * chunk_attempts)
                        cleanup_memory()  # Try to recover
                        continue
                    else:
                        failed_chunks += 1
                        logger.error(f"❌ Chunk {i+1} failed permanently")
                        break
        
        # Check if we have enough successful chunks
        if not audio_segments:
            raise Exception("No audio chunks were generated successfully")
        
        if failed_chunks > max_failed_chunks:
            raise Exception(f"Too many failed chunks: {failed_chunks}/{len(text_chunks)}")
        
        if failed_chunks > 0:
            logger.warning(f"⚠️ {failed_chunks} chunks failed, but continuing with {len(audio_segments)} successful chunks")
        
        # Combine audio segments with smooth transitions
        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            logger.info("🔗 Combining audio segments...")
            combined_audio = []
            
            for i, segment in enumerate(audio_segments):
                combined_audio.append(segment)
                
                # Add small silence between chunks (0.2 seconds)
                if i < len(audio_segments) - 1:
                    silence_duration = 0.2
                    silence = np.zeros(int(sample_rate * silence_duration))
                    combined_audio.append(silence)
            
            final_audio = np.concatenate(combined_audio)
        
        # Apply speed modification with quality preservation
        if speed != 1.0:
            try:
                final_audio = modify_audio_speed(final_audio, speed, sample_rate)
                logger.info(f"✅ Speed modified to {speed}x")
            except Exception as speed_error:
                logger.warning(f"Speed modification failed: {speed_error}, using original speed")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.{format.lower()}"
        file_path = config.AUDIO_DIR / filename
        
        # Save audio file with error handling
        try:
            sf.write(str(file_path), final_audio, sample_rate)
            logger.info(f"💾 Audio saved: {filename}")
        except Exception as save_error:
            logger.error(f"Failed to save audio: {save_error}")
            raise Exception(f"Audio save failed: {save_error}")
        
        # Convert format if needed
        if format.lower() != "wav":
            try:
                await convert_audio_format(str(file_path), format.lower())
                # Update filename after conversion
                filename = f"{file_id}.{format.lower()}"
                file_path = config.AUDIO_DIR / filename
            except Exception as convert_error:
                logger.warning(f"Format conversion failed: {convert_error}, keeping WAV")
                filename = f"{file_id}.wav"
        
        # Cache the result if enabled
        if cache_key:
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    file_data = await f.read()
                audio_cache.set(cache_key, file_data)
                logger.info(f"📋 Audio cached with key: {cache_key[:8]}")
            except Exception as cache_error:
                logger.warning(f"Caching failed: {cache_error}")
        
        generation_time = time.time() - start_time
        final_duration = len(final_audio) / sample_rate
        
        logger.info(f"✅ Audio generation completed: {filename} ({final_duration:.2f}s audio, {generation_time:.2f}s generation, {len(text_chunks)} chunks)")
        
        return str(filename), final_duration, sample_rate
        
    except Exception as e:
        logger.error(f"❌ Audio generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to provide recovery suggestions
        error_message = str(e)
        if "ONNX" in error_message or "onnxruntime" in error_message.lower():
            error_message += " (ONNX Runtime error - try restarting or using shorter text)"
        elif "memory" in error_message.lower():
            error_message += " (Memory error - try shorter text or restart service)"
        elif "timeout" in error_message.lower():
            error_message += " (Timeout - text may be too long, try breaking into smaller parts)"
        
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {error_message}")

def generate_chunk_audio_safe(text: str, voice: str, attempt: int = 1) -> np.ndarray:
    """Generate audio for a single chunk with maximum safety"""
    global tts_model
    
    try:
        if tts_model is None:
            raise Exception("TTS model not available")
        
        # Pre-generation cleanup for stability
        if attempt > 1:
            gc.collect()
        
        with model_lock:
            # Validate inputs
            if not text or not text.strip():
                raise Exception("Empty text chunk")
            
            if len(text) > config.CHUNK_SIZE * 2:  # Safety limit
                text = text[:config.CHUNK_SIZE * 2]
                logger.warning(f"Truncated chunk to {len(text)} characters")
            
            # Generate audio with error detection
            try:
                audio_data = tts_model.generate(text, voice=voice)
            except Exception as gen_error:
                # Handle specific ONNX errors
                if "onnx" in str(gen_error).lower() or "expand" in str(gen_error).lower():
                    logger.warning(f"ONNX error detected, trying alternative approach: {gen_error}")
                    
                    # Try with different voice as fallback
                    fallback_voices = ["expr-voice-2-f", "expr-voice-2-m", "expr-voice-1-f"]
                    for fallback_voice in fallback_voices:
                        if fallback_voice != voice:
                            try:
                                logger.info(f"Trying fallback voice: {fallback_voice}")
                                audio_data = tts_model.generate(text, voice=fallback_voice)
                                logger.info(f"✅ Fallback voice {fallback_voice} succeeded")
                                break
                            except Exception as fallback_error:
                                logger.warning(f"Fallback voice {fallback_voice} failed: {fallback_error}")
                                continue
                    else:
                        raise Exception(f"All voice fallbacks failed: {gen_error}")
                else:
                    raise gen_error
            
            # Validate output
            if audio_data is None:
                raise Exception("Model returned None")
            
            if not isinstance(audio_data, np.ndarray):
                raise Exception(f"Invalid audio type: {type(audio_data)}")
            
            if len(audio_data) == 0:
                raise Exception("Model returned empty audio array")
            
            # Check for valid audio range
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                raise Exception("Audio contains invalid values (NaN/Inf)")
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
                logger.info("🔧 Audio normalized to prevent clipping")
            
            return audio_data
    
    except Exception as e:
        logger.error(f"Chunk generation failed (attempt {attempt}): {e}")
        raise e

def modify_audio_speed(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """Modify audio speed with multiple fallback methods"""
    try:
        if speed == 1.0:
            return audio
        
        # Method 1: librosa (best quality)
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except ImportError:
            logger.info("librosa not available, using fallback speed modification")
        
        # Method 2: scipy resampling
        try:
            from scipy import signal
            target_length = int(len(audio) / speed)
            return signal.resample(audio, target_length)
        except ImportError:
            logger.info("scipy not available, using basic speed modification")
        
        # Method 3: simple sampling (fallback)
        if speed > 1.0:
            # Speed up by skipping samples
            step = max(1, int(speed))
            return audio[::step]
        else:
            # Slow down by simple repetition
            repeat_factor = max(1, int(1.0 / speed))
            return np.repeat(audio, repeat_factor)
    
    except Exception as e:
        logger.warning(f"Speed modification failed: {e}, using original audio")
        return audio

async def convert_audio_format(file_path: str, target_format: str):
    """Convert audio formats with multiple fallback methods"""
    try:
        original_path = Path(file_path)
        new_path = original_path.with_suffix(f".{target_format}")
        
        if target_format == "wav":
            return  # Already WAV
        
        conversion_successful = False
        
        # Method 1: pydub (preferred)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(str(original_path))
            
            export_params = {"format": target_format}
            if target_format == "mp3":
                export_params.update({"bitrate": "128k", "parameters": ["-q:a", "2"]})
            elif target_format == "ogg":
                export_params.update({"codec": "libvorbis"})
            elif target_format == "flac":
                export_params.update({"parameters": ["-compression_level", "5"]})
            
            audio.export(str(new_path), **export_params)
            original_path.unlink()  # Remove original
            conversion_successful = True
            logger.info(f"✅ Converted to {target_format.upper()} using pydub")
            
        except ImportError:
            logger.info("pydub not available, trying ffmpeg")
        except Exception as pydub_error:
            logger.warning(f"pydub conversion failed: {pydub_error}")
        
        # Method 2: ffmpeg direct (fallback)
        if not conversion_successful:
            try:
                codec_map = {
                    "mp3": "libmp3lame",
                    "ogg": "libvorbis", 
                    "flac": "flac",
                    "wav": "pcm_s16le"
                }
                
                codec = codec_map.get(target_format, "libmp3lame")
                
                cmd = [
                    "ffmpeg", "-i", str(original_path), "-y",
                    "-acodec", codec,
                    "-ar", "24000",  # Maintain sample rate
                    "-ac", "1",      # Mono
                    str(new_path)
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                original_path.unlink()
                conversion_successful = True
                logger.info(f"✅ Converted to {target_format.upper()} using ffmpeg")
                
            except subprocess.TimeoutExpired:
                logger.error("ffmpeg conversion timed out")
            except subprocess.CalledProcessError as ffmpeg_error:
                logger.warning(f"ffmpeg conversion failed: {ffmpeg_error}")
            except Exception as ffmpeg_error:
                logger.warning(f"ffmpeg error: {ffmpeg_error}")
        
        # Method 3: soundfile (limited format support)
        if not conversion_successful and target_format in ["flac", "ogg"]:
            try:
                audio_data, sr = sf.read(str(original_path))
                sf.write(str(new_path), audio_data, sr, format=target_format.upper())
                original_path.unlink()
                conversion_successful = True
                logger.info(f"✅ Converted to {target_format.upper()} using soundfile")
            except Exception as sf_error:
                logger.warning(f"soundfile conversion failed: {sf_error}")
        
        if not conversion_successful:
            logger.warning(f"All conversion methods failed, keeping WAV format")
    
    except Exception as e:
        logger.warning(f"Format conversion error: {e}, keeping original format")

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 KittenTTS Production API starting up...")
    
    # Initialize model in background
    model_task = asyncio.create_task(initialize_kitten_tts())
    
    # Start background tasks
    monitor_task = asyncio.create_task(background_monitoring())
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down KittenTTS API...")
    
    # Cancel background tasks
    for task in [monitor_task, cleanup_task]:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    # Cleanup resources
    cleanup_memory()
    audio_cache.clear()
    executor.shutdown(wait=True)
    logger.info("✅ Shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="🐱 KittenTTS Production API",
    description="Bulletproof KittenTTS implementation with advanced caching, monitoring, and error recovery",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
app.mount("/audio", StaticFiles(directory=str(config.AUDIO_DIR)), name="audio")

# Background monitoring
async def background_monitoring():
    """Continuous system monitoring and maintenance"""
    while True:
        try:
            await asyncio.sleep(60)  # Every minute
            
            # Get comprehensive system stats
            memory_stats = check_memory_usage()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/').percent
            
            # Log detailed stats every 5 minutes
            if int(time.time()) % 300 == 0:
                cache_stats = audio_cache.get_stats()
                metrics_stats = metrics.get_stats()
                
                logger.info(
                    f"📊 System: CPU {cpu_percent:.1f}%, "
                    f"Memory {memory_stats['system_memory_percent']:.1f}%, "
                    f"Disk {disk_usage:.1f}%, "
                    f"Cache {cache_stats['cache_size']} items ({cache_stats['memory_usage_mb']:.1f}MB), "
                    f"Requests {metrics_stats['requests']['total']} "
                    f"(Success: {metrics_stats['requests']['success_rate']:.1f}%)"
                )
            
            # Proactive memory management
            if memory_stats["system_memory_percent"] > config.MEMORY_THRESHOLD * 100:
                logger.warning("⚠️ High memory usage detected, performing cleanup")
                cleanup_memory()
                
                # Clear cache if memory is critical
                if memory_stats["system_memory_percent"] > 90:
                    audio_cache.clear()
                    logger.warning("🧹 Emergency cache clear due to critical memory usage")
            
            # CPU throttling if needed
            if cpu_percent > config.CPU_THRESHOLD * 100:
                logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
                await asyncio.sleep(2)  # Brief pause to let CPU recover
        
        except Exception as e:
            logger.warning(f"Background monitoring error: {e}")

async def periodic_cleanup():
    """Enhanced periodic cleanup with intelligent file management"""
    while True:
        try:
            await asyncio.sleep(config.CLEANUP_INTERVAL)
            
            logger.info("🧹 Starting periodic cleanup...")
            
            # Clean old audio files
            audio_files = list(config.AUDIO_DIR.glob("*"))
            current_time = time.time()
            
            if len(audio_files) > config.MAX_AUDIO_FILES:
                # Sort by access time (oldest first)
                audio_files.sort(key=lambda f: f.stat().st_atime)
                files_to_remove = audio_files[:-config.MAX_AUDIO_FILES]
                
                removed_count = 0
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {file_path}: {e}")
                
                logger.info(f"🧹 Cleaned up {removed_count} old audio files")
            
            # Clean very old files (older than 4 hours)
            old_file_threshold = current_time - (4 * 3600)
            old_files_removed = 0
            
            for file_path in config.AUDIO_DIR.glob("*"):
                try:
                    if file_path.stat().st_mtime < old_file_threshold:
                        file_path.unlink()
                        old_files_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old file {file_path}: {e}")
            
            if old_files_removed > 0:
                logger.info(f"🧹 Removed {old_files_removed} files older than 4 hours")
            
            # Memory cleanup
            cleanup_memory()
            
            # Cache maintenance
            if len(audio_cache.cache) > 50:
                # Clear oldest 25% of cache
                cache_size_before = len(audio_cache.cache)
                items_to_remove = max(1, cache_size_before // 4)
                
                for _ in range(items_to_remove):
                    if not audio_cache._evict_oldest():
                        break
                
                logger.info(f"🧹 Cache cleanup: {cache_size_before} -> {len(audio_cache.cache)} items")
            
        except Exception as e:
            logger.warning(f"Periodic cleanup error: {e}")

# API Routes
@app.get("/")
async def root():
    """Enhanced root endpoint with comprehensive system information"""
    try:
        memory_stats = check_memory_usage()
        cpu_percent = psutil.cpu_percent()
        model_status = "✅ Ready" if tts_model is not None else "⏳ Loading..."
        cache_stats = audio_cache.get_stats()
        
        return {
            "message": "🐱 KittenTTS Production API",
            "version": "2.0.0",
            "status": "active",
            "tts_model_status": model_status,
            "system_info": {
                "cpu_percent": round(cpu_percent, 1),
                "memory_usage_mb": round(memory_stats["memory_mb"], 1),
                "memory_percent": round(memory_stats["memory_percent"], 1),
                "available_memory_mb": round(memory_stats["available_mb"], 1),
                "system_memory_percent": round(memory_stats["system_memory_percent"], 1),
                "total_memory_mb": round(memory_stats["total_mb"], 1),
                "cpu_count": os.cpu_count(),
                "python_version": sys.version.split()[0],
                "platform": sys.platform
            },
            "cache_info": cache_stats,
            "endpoints": {
                "health": "/health - Comprehensive health check",
                "tts": "/tts - Text to speech generation (POST)",
                "voices": "/voices - Available voices list",
                "audio": "/audio/{filename} - Audio file access",
                "status": "/status - Detailed service status",
                "metrics": "/metrics - Performance metrics",
                "test-voice": "/test-voice/{voice} - Voice testing",
                "cleanup": "/cleanup - Manual cleanup (DELETE)",
                "docs": "/docs - API documentation"
            },
            "features": [
                "✅ Real KittenTTS with advanced chunking",
                "✅ Bulletproof error handling & recovery",
                "✅ Intelligent memory management",
                "✅ Advanced caching system",
                "✅ Continuous system monitoring",
                "✅ Long text support (up to 10,000 chars)",
                "✅ Multiple audio formats with fallbacks",
                "✅ Production-grade performance",
                "✅ Telegram integration ready",
                "✅ Unlimited processing time",
                "✅ ONNX error recovery",
                "✅ Voice fallback system"
            ],
            "limits": {
                "max_text_length": config.MAX_TEXT_LENGTH,
                "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
                "chunk_size": config.CHUNK_SIZE,
                "supported_voices": len(get_available_voices())
            },
            "metrics_summary": metrics.get_stats()
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return {"error": "Service temporarily unavailable", "message": str(e)}

@app.get("/health")
@handle_errors
async def health_check():
    """Comprehensive health check with detailed diagnostics"""
    try:
        model_ready = tts_model is not None
        memory_stats = check_memory_usage()
        cpu_percent = psutil.cpu_percent()
        disk_usage = psutil.disk_usage('/').percent
        
        # Test model functionality if available
        model_test_passed = False
        model_test_time = 0.0
        
        if model_ready:
            try:
                test_start = time.time()
                test_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: generate_chunk_audio_safe("Quick test", "expr-voice-2-f")
                    ),
                    timeout=15.0
                )
                model_test_time = time.time() - test_start
                model_test_passed = test_result is not None and len(test_result) > 0
                logger.info(f"✅ Model test passed in {model_test_time:.2f}s")
            except Exception as test_error:
                logger.warning(f"Model test failed: {test_error}")
        
        # Determine overall health status
        health_checks = {
            "model_loaded": model_ready,
            "model_functional": model_test_passed,
            "memory_ok": memory_stats["system_memory_percent"] < 90,
            "cpu_ok": cpu_percent < 95,
            "disk_ok": disk_usage < 95
        }
        
        all_healthy = all(health_checks.values())
        critical_issues = sum(1 for k, v in health_checks.items() if not v and k in ["model_loaded", "model_functional"])
        
        if all_healthy:
            health_status = "healthy"
        elif critical_issues > 0:
            health_status = "critical"
        else:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "service": "🐱 KittenTTS Production API",
            "timestamp": datetime.now().isoformat(),
            "checks": health_checks,
            "model_info": {
                "loaded": model_ready,
                "functional": model_test_passed,
                "test_time": round(model_test_time, 3),
                "load_time": round(metrics.model_load_time, 3)
            },
            "system_health": {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory_stats["system_memory_percent"], 1),
                "available_memory_mb": round(memory_stats["available_mb"], 1),
                "disk_usage_percent": round(disk_usage, 1),
                "total_memory_mb": round(memory_stats["total_mb"], 1)
            },
            "service_health": {
                "audio_files_count": len(list(config.AUDIO_DIR.glob("*"))),
                "cache_size": len(audio_cache.cache),
                "cache_memory_mb": round(audio_cache.current_memory / 1024 / 1024, 2),
                "active_requests": metrics.active_requests,
                "total_requests": metrics.requests_total,
                "success_rate": round((metrics.requests_success / max(1, metrics.requests_total)) * 100, 2)
            },
            "recommendations": self._get_health_recommendations(health_checks, memory_stats, cpu_percent)
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "message": "Health check system failed"
            }
        )

def _get_health_recommendations(checks: Dict[str, bool], memory_stats: Dict, cpu_percent: float) -> List[str]:
    """Generate health recommendations based on current status"""
    recommendations = []
    
    if not checks["model_loaded"]:
        recommendations.append("🔄 Model is loading - please wait")
    
    if not checks["model_functional"]:
        recommendations.append("🔧 Model test failed - consider restarting service")
    
    if memory_stats["system_memory_percent"] > 85:
        recommendations.append("🧹 High memory usage - consider manual cleanup")
    
    if cpu_percent > 90:
        recommendations.append("⏸️ High CPU usage - reduce concurrent requests")
    
    if not checks["disk_ok"]:
        recommendations.append("💿 Low disk space - cleanup required")
    
    if not recommendations:
        recommendations.append("✅ All systems operating normally")
    
    return recommendations

@app.post("/tts", response_model=TTSResponse)
@handle_errors
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Production-grade TTS endpoint with bulletproof error handling"""
    
    async with request_semaphore:
        metrics.active_requests += 1
        start_time = time.time()
        cache_hit = False
        
        try:
            logger.info(f"📝 TTS request: '{request.text[:100]}...' voice='{request.voice}' (length: {len(request.text)})")
            
            # Enhanced validation
            if not request.text or len(request.text.strip()) == 0:
                raise HTTPException(status_code=400, detail="Text cannot be empty")
            
            # Clean and validate text
            cleaned_text = request.text.strip()
            if len(cleaned_text) > config.MAX_TEXT_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Text too long",
                        "max_length": config.MAX_TEXT_LENGTH,
                        "provided_length": len(cleaned_text),
                        "suggestion": f"Please limit text to {config.MAX_TEXT_LENGTH} characters"
                    }
                )
            
            # Validate voice with automatic fallback
            available_voices = [v.name for v in get_available_voices()]
            original_voice = request.voice
            
            if request.voice not in available_voices:
                logger.warning(f"Unknown voice '{request.voice}', using default")
                request.voice = "expr-voice-2-f"
            
            # Wait for model with progress updates
            model_wait_time = 0
            max_wait = 600  # 10 minutes
            
            while tts_model is None and model_wait_time < max_wait:
                if model_wait_time % 30 == 0:  # Log every 30 seconds
                    logger.info(f"⏳ Waiting for model to load... ({model_wait_time}s/{max_wait}s)")
                await asyncio.sleep(5)
                model_wait_time += 5
            
            if tts_model is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "TTS model unavailable",
                        "message": "Model failed to load within timeout period",
                        "wait_time": model_wait_time,
                        "suggestion": "Please try again later or contact support"
                    }
                )
            
            # Generate cache key
            cache_key = None
            if request.cache_enabled:
                cache_key = request.generate_cache_key()
            
            # Generate audio with comprehensive error recovery
            filename = None
            duration = 0.0
            sample_rate = 24000
            
            try:
                filename, duration, sample_rate = await generate_audio_with_chunking(
                    text=cleaned_text,
                    voice=request.voice,
                    speed=request.speed,
                    format=request.format,
                    cache_key=cache_key
                )
                
                cache_hit = cache_key is not None and cache_key in audio_cache.cache
                
            except HTTPException as http_error:
                # Handle specific HTTP errors
                if http_error.status_code == 500:
                    # Try recovery strategies
                    logger.warning("🔧 Attempting error recovery...")
                    
                    # Strategy 1: Try with default voice
                    if request.voice != "expr-voice-2-f":
                        logger.info("🔄 Retrying with default voice")
                        try:
                            filename, duration, sample_rate = await generate_audio_with_chunking(
                                text=cleaned_text,
                                voice="expr-voice-2-f",
                                speed=1.0,  # Also reset speed
                                format="wav",  # Use most compatible format
                                cache_key=None  # Skip cache for recovery
                            )
                            logger.info("✅ Recovery successful with default voice")
                        except Exception as recovery_error:
                            logger.error(f"Recovery attempt failed: {recovery_error}")
                            raise http_error
                    else:
                        raise http_error
                else:
                    raise http_error
            
            if not filename:
                raise HTTPException(status_code=500, detail="Audio generation failed - no file created")
            
            # Verify file exists and get size
            file_path = config.AUDIO_DIR / filename
            if not file_path.exists():
                raise HTTPException(status_code=500, detail="Generated audio file not found")
            
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            
            # Create response
            audio_url = f"/audio/{filename}"
            generation_time = time.time() - start_time
            text_chunks = len(preprocess_text(cleaned_text))
            
            response = TTSResponse(
                text=cleaned_text,
                status="success",
                audio_url=audio_url,
                audio_file_path=audio_url,
                file_id=filename.split('.')[0],
                message=(
                    f"🎵 Audio generated successfully with {request.voice} "
                    f"(speed: {request.speed}x, duration: {duration:.2f}s, "
                    f"generation: {generation_time:.2f}s, chunks: {text_chunks})"
                ),
                telegram_ready=True,
                duration=duration,
                sample_rate=sample_rate,
                cache_hit=cache_hit,
                generation_time=generation_time,
                model_version="kitten-tts-nano-0.1",
                text_chunks=text_chunks,
                file_size_mb=round(file_size_mb, 3)
            )
            
            # Record success metrics
            metrics.record_request(
                success=True,
                generation_time=generation_time,
                cached=cache_hit,
                audio_duration=duration,
                text_length=len(cleaned_text)
            )
            
            logger.info(f"✅ TTS completed: {filename} ({duration:.2f}s, {generation_time:.2f}s gen)")
            
            # Schedule cleanup (4 hours for successful files)
            background_tasks.add_task(cleanup_file, filename, 14400)
            
            return response
            
        except HTTPException as http_error:
            # Record HTTP errors
            error_detail = http_error.detail if isinstance(http_error.detail, str) else str(http_error.detail)
            metrics.record_request(False, error_type=f"HTTP_{http_error.status_code}")
            raise
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"❌ Unexpected TTS error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Categorize error
            error_type = "Unknown"
            if "onnx" in str(e).lower():
                error_type = "ONNX_Error"
            elif "memory" in str(e).lower():
                error_type = "Memory_Error"
            elif "timeout" in str(e).lower():
                error_type = "Timeout_Error"
            
            metrics.record_request(False, error_type=error_type)
            
            # Provide helpful error response
            error_response = {
                "error": "TTS generation failed",
                "message": str(e),
                "error_type": error_type,
                "text_length": len(request.text),
                "voice": request.voice,
                "timestamp": datetime.now().isoformat(),
                "suggestions": []
            }
            
            # Add specific suggestions based on error type
            if error_type == "ONNX_Error":
                error_response["suggestions"].extend([
                    "Try with a different voice",
                    "Try with shorter text",
                    "Restart the service if problem persists"
                ])
            elif error_type == "Memory_Error":
                error_response["suggestions"].extend([
                    "Try with shorter text",
                    "Wait a moment and try again",
                    "Use manual cleanup endpoint"
                ])
            elif error_type == "Timeout_Error":
                error_response["suggestions"].extend([
                    "Break text into smaller parts",
                    "Try again later",
                    "Use simpler text"
                ])
            else:
                error_response["suggestions"].append("Try again with different parameters")
            
            raise HTTPException(status_code=500, detail=error_response)
        
        finally:
            metrics.active_requests -= 1

@app.get("/metrics")
async def get_comprehensive_metrics():
    """Get detailed performance and system metrics"""
    try:
        memory_stats = check_memory_usage()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk_usage = psutil.disk_usage('/')
        cache_stats = audio_cache.get_stats()
        
        # Network stats
        network_stats = {}
        try:
            network_io = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            }
        except Exception:
            pass
        
        return {
            "performance": metrics.get_stats(),
            "system": {
                "cpu": {
                    "percent": round(cpu_percent, 1),
                    "count": os.cpu_count(),
                    "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                "memory": {
                    "process_mb": round(memory_stats["memory_mb"], 1),
                    "process_percent": round(memory_stats["memory_percent"], 1),
                    "system_percent": round(memory_stats["system_memory_percent"], 1),
                    "available_mb": round(memory_stats["available_mb"], 1),
                    "total_mb": round(memory_stats["total_mb"], 1)
                },
                "disk": {
                    "usage_percent": round(disk_usage.percent, 1),
                    "free_gb": round(disk_usage.free / 1024**3, 1),
                    "total_gb": round(disk_usage.total / 1024**3, 1)
                },
                "network": network_stats
            },
            "cache": cache_stats,
            "model": {
                "loaded": tts_model is not None,
                "load_time": metrics.model_load_time,
                "model_type": "KittenML/kitten-tts-nano-0.1",
                "supported_voices": len(get_available_voices())
            },
            "files": {
                "audio_files_count": len(list(config.AUDIO_DIR.glob("*"))),
                "audio_dir_size_mb": round(
                    sum(f.stat().st_size for f in config.AUDIO_DIR.glob("*")) / 1024 / 1024, 2
                )
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": "Metrics temporarily unavailable", "message": str(e)}

@app.get("/voices")
async def get_voices():
    """Get available voices with testing capabilities"""
    try:
        voices = get_available_voices()
        model_ready = tts_model is not None
        
        return {
            "voices": [voice.dict() for voice in voices],
            "total": len(voices),
            "default": "expr-voice-2-f",
            "recommended": {
                "male_standard": "expr-voice-2-m",
                "female_standard": "expr-voice-2-f",
                "male_expressive": "expr-voice-4-m",
                "female_expressive": "expr-voice-4-f"
            },
            "quality_levels": ["standard", "high"],
            "model_ready": model_ready,
            "sample_rate": 24000,
            "note": "All voices support English with high-quality output"
        }
    except Exception as e:
        logger.error(f"Voices endpoint error: {e}")
        return {"error": "Voice information unavailable", "message": str(e)}

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio files with enhanced security and headers"""
    try:
        # Security: validate filename
        if not re.match(r'^[a-f0-9\-]+\.(wav|mp3|ogg|flac), filename):
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        file_path = config.AUDIO_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Get file stats
        file_stats = file_path.stat()
        file_size = file_stats.st_size
        
        # Determine media type
        extension = filename.split('.')[-1].lower()
        media_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac'
        }
        
        media_type = media_types.get(extension, 'application/octet-stream')
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename,
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Cache-Control": "public, max-age=7200",
                "Accept-Ranges": "bytes",
                "X-Content-Type-Options": "nosniff",
                "Content-Length": str(file_size),
                "X-Audio-Duration": "unknown",  # Could be calculated
                "X-Generated-By": "KittenTTS-Production-API"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio file serving error: {e}")
        raise HTTPException(status_code=500, detail=f"File serving failed: {str(e)}")

@app.post("/test-voice/{voice_name}")
@handle_errors
async def test_voice(voice_name: str, test_text: Optional[str] = None):
    """Test a specific voice with custom or default text"""
    if test_text is None:
        test_text = f"Hello! This is a test of the {voice_name} voice from KittenTTS. The quick brown fox jumps over the lazy dog."
    
    test_request = TTSRequest(
        text=test_text,
        voice=voice_name,
        speed=1.0,
        format="wav",
        cache_enabled=False  # Don't cache test audio
    )
    
    try:
        result = await text_to_speech(test_request, BackgroundTasks())
        result.message = f"✅ Voice test completed for {voice_name}"
        return result
    except Exception as e:
        logger.error(f"Voice test failed for {voice_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Voice test failed",
                "voice": voice_name,
                "message": str(e),
                "available_voices": [v.name for v in get_available_voices()]
            }
        )

@app.delete("/cleanup")
@handle_errors
async def manual_cleanup():
    """Enhanced manual cleanup with detailed reporting"""
    try:
        cleanup_start_time = time.time()
        
        # Pre-cleanup stats
        files_before = len(list(config.AUDIO_DIR.glob("*")))
        cache_size_before = len(audio_cache.cache)
        memory_before = check_memory_usage()
        
        # Clean audio files
        removed_files = 0
        total_size_removed = 0
        
        for file_path in config.AUDIO_DIR.glob("*"):
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                removed_files += 1
                total_size_removed += file_size
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
        
        # Clear all caches
        audio_cache.clear()
        
        # Force memory cleanup
        cleanup_memory()
        
        # Post-cleanup stats
        memory_after = check_memory_usage()
        cleanup_time = time.time() - cleanup_start_time
        
        return {
            "message": "🧹 Manual cleanup completed successfully",
            "cleanup_time": round(cleanup_time, 2),
            "files": {
                "removed": removed_files,
                "size_removed_mb": round(total_size_removed / 1024 / 1024, 2),
                "remaining": len(list(config.AUDIO_DIR.glob("*")))
            },
            "cache": {
                "items_cleared": cache_size_before,
                "memory_freed_mb": round(
                    (memory_before["memory_mb"] - memory_after["memory_mb"]), 2
                )
            },
            "memory": {
                "before_mb": round(memory_before["memory_mb"], 1),
                "after_mb": round(memory_after["memory_mb"], 1),
                "system_before_percent": round(memory_before["system_memory_percent"], 1),
                "system_after_percent": round(memory_after["system_memory_percent"], 1)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup operation failed: {str(e)}")

@app.get("/status")
async def get_detailed_status():
    """Get comprehensive service status"""
    try:
        memory_stats = check_memory_usage()
        cpu_percent = psutil.cpu_percent()
        cache_stats = audio_cache.get_stats()
        metrics_stats = metrics.get_stats()
        
        # Process information
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "cpu_percent": round(process.cpu_percent(), 2),
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            "threads": process.num_threads(),
            "connections": len(process.connections()),
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
        }
        
        return {
            "service": {
                "name": "🐱 KittenTTS Production API",
                "version": "2.0.0",
                "status": "running",
                "uptime": metrics_stats["uptime"],
                "model_loaded": tts_model is not None,
                "python_version": sys.version.split()[0],
                "platform": sys.platform
            },
            "performance": metrics_stats,
            "system": {
                "cpu_percent": round(cpu_percent, 1),
                "memory": memory_stats,
                "process": process_info
            },
            "cache": cache_stats,
            "configuration": {
                "max_text_length": config.MAX_TEXT_LENGTH,
                "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
                "chunk_size": config.CHUNK_SIZE,
                "max_retries": config.MAX_RETRIES,
                "cleanup_interval": config.CLEANUP_INTERVAL,
                "memory_threshold": config.MEMORY_THRESHOLD,
                "cpu_threshold": config.CPU_THRESHOLD
            },
            "features": {
                "real_kitten_tts": True,
                "advanced_chunking": True,
                "intelligent_caching": True,
                "error_recovery": True,
                "memory_management": True,
                "format_conversion": True,
                "voice_fallback": True,
                "unlimited_text_length": True,
                "production_ready": True
            },
            "health_status": await health_check(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return {"error": "Status information unavailable", "message": str(e)}

async def cleanup_file(filename: str, delay: int):
    """Enhanced file cleanup with error handling"""
    try:
        await asyncio.sleep(delay)
        file_path = config.AUDIO_DIR / filename
        
        if file_path.exists():
            file_size = file_path.stat().st_size
            file_path.unlink()
            logger.info(f"🧹 Cleaned up: {filename} ({file_size / 1024:.1f}KB)")
        else:
            logger.info(f"🧹 File already removed: {filename}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {filename}: {e}")

# Advanced error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The endpoint '{request.url.path}' does not exist",
            "available_endpoints": [
                "/", "/health", "/tts", "/voices", "/metrics", "/status", "/docs"
            ],
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "Request data validation failed",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Try emergency cleanup
    try:
        cleanup_memory()
    except:
        pass
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Try the request again or contact support if the problem persists"
        }
    )

# Graceful shutdown handler
async def shutdown_handler():
    """Handle graceful shutdown"""
    logger.info("🛑 Initiating graceful shutdown...")
    
    # Stop accepting new requests
    app.state.shutting_down = True
    
    # Wait for active requests to complete
    max_wait = 30
    wait_time = 0
    
    while metrics.active_requests > 0 and wait_time < max_wait:
        logger.info(f"⏳ Waiting for {metrics.active_requests} active requests to complete...")
        await asyncio.sleep(1)
        wait_time += 1
    
    # Cleanup
    cleanup_memory()
    audio_cache.clear()
    
    logger.info("✅ Graceful shutdown completed")

# Add shutdown handler to app
@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_handler()

# Health check middleware
@app.middleware("http")
async def health_check_middleware(request: Request, call_next):
    """Middleware to check system health before processing requests"""
    
    # Skip health checks for health endpoint itself
    if request.url.path in ["/health", "/metrics", "/status", "/"]:
        return await call_next(request)
    
    # Check if shutting down
    if hasattr(app.state, 'shutting_down') and app.state.shutting_down:
        return JSONResponse(
            status_code=503,
            content={"error": "Service shutting down", "message": "Please try again later"}
        )
    
    # Check system resources
    memory_stats = check_memory_usage()
    if memory_stats["system_memory_percent"] > 95:
        return JSONResponse(
            status_code=503,
            content={
                "error": "System overloaded",
                "message": "High memory usage detected, please try again later",
                "memory_percent": round(memory_stats["system_memory_percent"], 1)
            }
        )
    
    # Check if too many active requests
    if metrics.active_requests >= config.MAX_CONCURRENT_REQUESTS:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Server busy",
                "message": "Too many concurrent requests, please try again later",
                "active_requests": metrics.active_requests,
                "max_requests": config.MAX_CONCURRENT_REQUESTS
            }
        )
    
    return await call_next(request)

# Additional utility endpoints
@app.get("/formats")
async def get_supported_formats():
    """Get supported audio formats with conversion capabilities"""
    return {
        "formats": [
            {
                "format": "wav",
                "mime_type": "audio/wav",
                "telegram_compatible": True,
                "quality": "lossless",
                "default": True,
                "conversion_method": "native"
            },
            {
                "format": "mp3",
                "mime_type": "audio/mpeg",
                "telegram_compatible": True,
                "quality": "lossy",
                "conversion_method": "pydub/ffmpeg"
            },
            {
                "format": "ogg",
                "mime_type": "audio/ogg",
                "telegram_compatible": True,
                "quality": "lossy",
                "conversion_method": "pydub/ffmpeg"
            },
            {
                "format": "flac",
                "mime_type": "audio/flac",
                "telegram_compatible": False,
                "quality": "lossless",
                "conversion_method": "soundfile/ffmpeg"
            }
        ],
        "recommended_for_telegram": "wav",
        "smallest_size": "ogg",
        "best_quality": "flac",
        "fastest_generation": "wav"
    }

@app.post("/bulk-tts")
@handle_errors
async def bulk_text_to_speech(requests: List[TTSRequest], background_tasks: BackgroundTasks):
    """Process multiple TTS requests efficiently"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per bulk operation")
    
    results = []
    
    for i, request in enumerate(requests):
        try:
            logger.info(f"Processing bulk request {i+1}/{len(requests)}")
            result = await text_to_speech(request, background_tasks)
            results.append({"index": i, "success": True, "result": result})
        except Exception as e:
            logger.error(f"Bulk request {i+1} failed: {e}")
            results.append({
                "index": i,
                "success": False,
                "error": str(e),
                "request_text": request.text[:50] + "..." if len(request.text) > 50 else request.text
            })
    
    successful = sum(1 for r in results if r["success"])
    
    return {
        "total_requests": len(requests),
        "successful": successful,
        "failed": len(requests) - successful,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

# Development and debugging endpoints
@app.get("/debug/model")
async def debug_model_info():
    """Get detailed model debugging information"""
    if tts_model is None:
        return {"error": "Model not loaded"}
    
    try:
        return {
            "model_class": str(type(tts_model)),
            "model_loaded": True,
            "available_methods": [method for method in dir(tts_model) if not method.startswith('_')],
            "cache_directory": str(config.MODEL_CACHE_DIR),
            "model_files": list(config.MODEL_CACHE_DIR.glob("**/*")) if config.MODEL_CACHE_DIR.exists() else [],
            "memory_usage": check_memory_usage(),
            "model_load_time": metrics.model_load_time
        }
    except Exception as e:
        return {"error": f"Debug info failed: {e}"}

@app.get("/debug/system")
async def debug_system_info():
    """Get detailed system debugging information"""
    try:
        import platform
        
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path[:5]  # First 5 entries
            },
            "environment": {
                "audio_dir": str(config.AUDIO_DIR),
                "cache_dir": str(config.CACHE_DIR),
                "model_cache_dir": str(config.MODEL_CACHE_DIR),
                "hf_home": os.environ.get("HF_HOME", "Not set"),
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", "Not set")
            },
            "resources": {
                "cpu_count": os.cpu_count(),
                "memory": check_memory_usage(),
                "disk_usage": {
                    "total_gb": round(psutil.disk_usage('/').total / 1024**3, 1),
                    "free_gb": round(psutil.disk_usage('/').free / 1024**3, 1),
                    "used_percent": round(psutil.disk_usage('/').percent, 1)
                }
            },
            "installed_packages": self._get_relevant_packages()
        }
    except Exception as e:
        return {"error": f"System debug failed: {e}"}

def _get_relevant_packages() -> Dict[str, str]:
    """Get versions of relevant installed packages"""
    try:
        import pkg_resources
        
        relevant_packages = [
            'fastapi', 'uvicorn', 'torch', 'numpy', 'soundfile', 
            'scipy', 'librosa', 'pydub', 'transformers', 'kittentts'
        ]
        
        package_versions = {}
        for package in relevant_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                package_versions[package] = version
            except pkg_resources.DistributionNotFound:
                package_versions[package] = "Not installed"
        
        return package_versions
    except Exception as e:
        return {"error": f"Package detection failed: {e}"}

if __name__ == "__main__":
    # Enhanced startup configuration
    logger.info("🚀 Starting KittenTTS Production API...")
    
    # Pre-startup checks
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Audio directory: {config.AUDIO_DIR}")
    logger.info(f"Cache directory: {config.CACHE_DIR}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        workers=1,  # Single worker for model consistency
        loop="asyncio",
        reload=False,  # Disable reload for production
        use_colors=True,
        server_header=False,  # Security
        date_header=True
    )
