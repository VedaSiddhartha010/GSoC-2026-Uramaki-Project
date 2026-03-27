"""
Chunk-level Voice Analysis Service
Aligns voice features with transcript timestamps
"""

from pydub import AudioSegment

from app.config import Config
from app.services.voice_analysis_service import VoiceAnalysisService
from app.utils.logger import logger


class VoiceChunkAnalysisService:
    def __init__(self):
        config = Config().config
        self.voice_service = VoiceAnalysisService(config)

    def analyze_chunks(self, audio_path: str, chunks: list) -> list:
        """
        Perform voice analysis aligned with transcript chunks
        """
        try:
            audio = AudioSegment.from_file(audio_path)

            enhanced_chunks = []

            for chunk in chunks:
                timestamp = chunk.get("timestamp", [0, 0])
                start_ms, end_ms = int(timestamp[0]), int(timestamp[1])

                # Extract audio segment
                segment = audio[start_ms:end_ms]

                # Temporary export (in-memory alternative avoided for simplicity)
                temp_path = f"static/temp_chunk_{start_ms}_{end_ms}.wav"
                segment.export(temp_path, format="wav")

                # Analyze segment
                voice_result = self.voice_service.analyze(temp_path)

                # Attach results
                emotion_data = voice_result.get("emotion", {})
                cognitive_data = voice_result.get("cognitive_state", {})

                chunk["voice_analysis"] = {
                 "emotion": emotion_data.get("label") if isinstance(emotion_data, dict) else emotion_data,
                 "cognitive_load": cognitive_data.get("cognitive_load")
            }
                enhanced_chunks.append(chunk)

            return enhanced_chunks

        except Exception as e:
            logger.error(f"[error] [VoiceChunkAnalysisService] {str(e)}")
            return chunks