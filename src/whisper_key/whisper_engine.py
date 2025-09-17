import logging
import time
import asyncio
from typing import Optional
import numpy as np
from urllib.parse import urlparse
from wyoming.asr import Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe


class WhisperEngine:
    """Wyoming protocol WhisperEngine.

    Client to Wyoming faster-whisper service (linuxserver/faster-whisper container).
    Uses Wyoming protocol for asynchronous communication over TCP.
    """

    SAMPLE_RATE = 16000

    def __init__(self,
                 base_url: str,
                 model_size: str = "base",
                 language: str | None = None,
                 beam_size: int = 5,
                 remote_model: str | None = None,
                 timeout: float = 30.0):
        # Parse URL to get host and port
        if not base_url.startswith(('tcp://', 'http://', 'https://')):
            base_url = f"tcp://{base_url}"
        
        parsed = urlparse(base_url)
        self.host = parsed.hostname or 'localhost'
        self.port = parsed.port or 10300
        self.model_size = model_size
        self.remote_model = remote_model or ""
        self.language = None if language == 'auto' else language
        self.beam_size = beam_size
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.device = 'remote'
        self.compute_type = 'remote'
        self._info_cache = None
        self._info_cache_ts = None

    # Backward compatibility
    def is_loading(self) -> bool:
        return False

    def change_model(self, new_model_size: str, progress_callback: Optional[callable] = None):
        old = self.model_size
        self.model_size = new_model_size
        if progress_callback:
            progress_callback(f"Switched remote target model {old} -> {new_model_size}")

    def update_server_url(self, new_url: str):
        """Updates server URL for Wyoming connection."""
        # Convert URL to host:port format for Wyoming
        if not new_url.startswith(('tcp://', 'http://', 'https://')):
            new_url = f"tcp://{new_url}"
        
        parsed = urlparse(new_url)
        old_host, old_port = self.host, self.port
        self.host = parsed.hostname or 'localhost'
        self.port = parsed.port or 10300
        
        # Reset cache when changing server
        self._info_cache = None
        self._info_cache_ts = None
        
        self.logger.info(f"Updated Wyoming server: {old_host}:{old_port} -> {self.host}:{self.port}")

    @property
    def base_url(self) -> str:
        """UI compatibility - returns URL in host:port format."""
        return f"{self.host}:{self.port}"
    
    @base_url.setter
    def base_url(self, url: str):
        """UI compatibility - sets URL through update_server_url."""
        self.update_server_url(url)

    def health_check(self) -> bool:
        """Checks Wyoming server availability through TCP connection."""
        try:
            # Simple TCP connection check
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.debug(f"Health check error: {e}")
            return False

    async def _get_info(self) -> dict | None:
        """Gets server information through Wyoming protocol."""
        try:
            client = AsyncTcpClient(self.host, self.port)
            
            # Establish connection
            await client.connect()
            
            # Send information request
            await client.write_event(Describe().event())
            
            # Wait for response
            event = await asyncio.wait_for(client.read_event(), timeout=5.0)
            await client.disconnect()
            
            if event and event.type == 'info':
                return event.data
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get Wyoming info: {e}")
            return None

    def get_models(self, max_age: float = 60.0) -> list[str] | None:
        """Get list of available models from Wyoming server. Caches result for max_age seconds."""
        now = time.time()
        if self._info_cache and self._info_cache_ts and (now - self._info_cache_ts) < max_age:
            # Extract models from cached information
            asr_info = self._info_cache.get('asr', [])
            if asr_info and len(asr_info) > 0:
                models = asr_info[0].get('models', [])
                return [model.get('name', '') for model in models if model.get('name')]
        
        try:
            # Run asynchronous function in synchronous context
            import asyncio
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            info = loop.run_until_complete(self._get_info())
            if info:
                self._info_cache = info
                self._info_cache_ts = now
                
                asr_info = info.get('asr', [])
                if asr_info and len(asr_info) > 0:
                    models = asr_info[0].get('models', [])
                    return [model.get('name', '') for model in models if model.get('name')]
            return None
        except Exception as e:
            self.logger.debug(f"Failed to fetch Wyoming models: {e}")
            return None

    def get_active_model(self) -> str | None:
        """Get currently configured model.
        Since faster-whisper ignores Wyoming model parameter,
        return locally configured model."""
        return self.remote_model if self.remote_model else self.model_size

    async def _transcribe_audio_async(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
        """Asynchronous transcription through Wyoming protocol."""
        try:
            # Connect to server
            client = AsyncTcpClient(self.host, self.port)
            await client.connect()
            
            # Convert audio to required format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            audio_clip = np.clip(audio_data, -1.0, 1.0)
            
            # Convert to int16 for Wyoming
            audio_int16 = (audio_clip * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Send transcription request
            transcribe_request = Transcribe(language=self.language).event()
            await client.write_event(transcribe_request)
            
            # Send audio data
            audio_start = AudioStart(
                rate=sample_rate,
                width=2,  # 16-bit = 2 bytes
                channels=1
            ).event()
            await client.write_event(audio_start)
            
            # Send audio in chunks
            chunk_size = 1024
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                audio_chunk = AudioChunk(
                    rate=sample_rate,
                    width=2,
                    channels=1,
                    audio=chunk
                ).event()
                await client.write_event(audio_chunk)
            
            # Finish audio stream
            audio_stop = AudioStop().event()
            await client.write_event(audio_stop)
            
            # Wait for transcription result
            while True:
                event = await asyncio.wait_for(client.read_event(), timeout=self.timeout)
                if event and event.type == 'transcript':
                    text = event.data.get('text', '')
                    await client.disconnect()
                    return text.strip() if text else None
                elif event and event.type == 'error':
                    error_msg = event.data.get('text', 'Unknown error')
                    self.logger.error(f"Wyoming transcription error: {error_msg}")
                    await client.disconnect()
                    return None
                    
        except Exception as e:
            self.logger.error(f"Wyoming transcription error: {e}")
            try:
                await client.disconnect()
            except Exception:
                pass
            return None

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe (Wyoming)")
            return None

        try:
            # Run asynchronous transcription in synchronous context
            import asyncio
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            start = time.time()
            result = loop.run_until_complete(self._transcribe_audio_async(audio_data, sample_rate))
            elapsed = time.time() - start
            
            if result:
                self.logger.info(f"Wyoming transcription done in {elapsed:.2f}s: '{result}'")
            return result
            
        except Exception as e:
            self.logger.error(f"Wyoming transcription error: {e}")
            return None
    