"""
Voice Analysis Service
Provides speech feature extraction, emotion detection,
and cognitive load estimation aligned with RUXAILAB pipeline.

 PATTERNS USED :---
- Type hints on all parameters and return values
- Instance variables with type annotations
- Comprehensive docstrings explaining every method
- Error handling with proper logging
"""
from app.models.whisper_model import WhisperTranscript
from app.models.speech_emotion_model import SpeechEmotionModel
from typing import Dict, Any
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence

from app.utils.logger import logger


class VoiceAnalysisService:
    """Service layer for voice/speech analysis operations.
    
    This class handles :---
    1. Speech feature extraction (energy, zero-crossing rate, silence detection).
    2. Emotion detection based on voice characteristics.
    3. Cognitive load estimation from speech patterns.
    
    All methods are type-hinted for better IDE support and type safety.
    """
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

        try:
            if self.config:
                from app.models.whisper_model import WhisperTranscript
                from app.models.speech_emotion_model import SpeechEmotionModel

                self.transcriber = WhisperTranscript(self.config)
                self.emotion_model = SpeechEmotionModel(self.config)
            else:
                self.transcriber = None
                self.emotion_model = None

        except Exception as e:
            print("TRANSCRIPTION ERROR:", e)
            logger.error(f"Failed to initialize models: {e}")
            self.transcriber = None
            self.emotion_model = None

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """Main entry point for voice analysis.
        
        TYPE ANNOTATIONS:
        - 'audio_path: str': Parameter must be a string (file path)
        - '-> Dict[str, Any]': Returns a dictionary with string keys and any values
        
        Args:
            audio_path (str): Path to the audio file to analyze
            
        Returns:
            Dict with analysis results:
            - 'speech_features': Dict of extracted speech features
            - 'emotion': Dict with emotion label
            - 'cognitive_state': Dict with cognitive load estimation
            - 'error': Error message if analysis fails
            
        process :--
        1. Load audio file as AudioSegment
        2. Extract speech features (energy, silence, etc.)
        3. Detect emotion based on features
        4. Estimate cognitive state
        5. Return all results combined
        """
        try:
            # Step 1: Load audio file
            audio: AudioSegment = AudioSegment.from_file(audio_path)

            # Step 2: Extract speech features
            features: Dict[str, Any] = self._extract_speech_features(audio)

            # Step 3: Detect emotion from features
            emotion: Dict[str, Any] = self._detect_emotion(features)
            
            # Step 4: Estimate cognitive state from features
            cognitive: Dict[str, Any] = self._estimate_cognitive_state(features)

            # --- NEW: transcription ---
            transcription = None
            chunks = []

            if self.transcriber:
                try:
                    res = self.transcriber(audio_path)
                    transcription = res.get("text")
                    chunks = res.get("chunks", [])

                    for chunk in chunks:
                        if isinstance(chunk.get("timestamp"), tuple):
                            chunk["timestamp"] = list(chunk["timestamp"])

                except Exception as e:
                    logger.error(f"[error] transcription failed: {str(e)}")

            # --- NEW: ML emotion ---
            ml_emotion = None

            if self.emotion_model:
                try:
                    ml_emotion = self.emotion_model(audio_path)
                except Exception as e:
                    logger.error(f"[error] emotion model failed: {str(e)}")

            # --- CLEAN EMOTION OUTPUT ---
            final_emotion = ml_emotion or emotion

            # Handle nested structure
            if isinstance(final_emotion, dict) and "emotion" in final_emotion:
                final_emotion = final_emotion["emotion"]

            # Normalize labels
            label_map = {
                "neu": "neutral",
                "ang": "anger",
                "hap": "happy",
                "sad": "sad",
                "fea": "fear"
            }

            if isinstance(final_emotion, dict):
                lbl = final_emotion.get("label", "")
                final_emotion["label"] = label_map.get(lbl, lbl)

            return {
                "transcription": transcription,           # NEW
                "chunks": chunks,                         # NEW
                "speech_features": features,
                "emotion": final_emotion,         # ML first, fallback to rule-based
                "cognitive_state": cognitive
            }

        except Exception as e:
            # Catch any unexpected errors and log them
            logger.error(f"[error] [VoiceAnalysisService] [analyze] {str(e)}")
            return {"error": "Voice analysis failed"}

    # ================= SPEECH FEATURES =================
    def _extract_speech_features(self, audio: AudioSegment) -> Dict[str, Any]:
        """Extract speech features from audio segment.
        
        TYPE ANNOTATIONS:
        - 'audio: AudioSegment': Parameter must be a pydub AudioSegment object
        - '-> Dict[str, Any]': Returns a dictionary with feature names and values
        
        Args:
            audio (AudioSegment): Audio segment to analyze
            
        Returns:
            Dict with features :--
            - 'duration_sec': Total audio duration in seconds (float)
            - 'energy': RMS energy level (float)
            - 'zcr': Zero-crossing rate indicating pitch/frequency changes (float)
            - 'silence_ratio': Proportion of silent sections (0-1, float)
            - 'speaking_ratio': Proportion of speech sections (0-1, float)
            - 'dynamic_range': Difference between loudest and quietest (float)
            
        TECHNICAL DETAILS :--
        - Energy: RMS (root mean square) of audio samples, high = loud speech
        - ZCR: Frequency of sign changes in waveform, high = consonants
        - Silence detection: Uses audio.dBFS - 14 as threshold (converted to int)
        """
        try:
            # Step 1: Convert audio to numpy array for analysis
            samples: np.ndarray = np.array(audio.get_array_of_samples())

            # Step 2: Calculate duration in seconds
            duration: float = len(audio) / 1000.0

            # Step 3: Calculate Energy (RMS - Root Mean Square)
            # RMS measures "loudness" by averaging the power
            energy = float(np.sqrt(np.mean(samples**2))) / (np.max(np.abs(samples)) + 1e-6) if len(samples) > 0 else 0.0

            # Step 4: Calculate Zero Crossing Rate (ZCR)
            # How often the waveform crosses zero axis
            # High ZCR = more high-frequency content (consonants, excitement)
            zero_crossings: float = np.sum(np.abs(np.diff(np.sign(samples)))) / len(samples) if len(samples) > 0 else 0.0

            # Step 5: Detect silence ranges
            # IMPORTANT: Convert 'audio.dBFS - 14' to int to match detect_silence() parameter type
            # audio.dBFS is float, -14 is int, result is float
            # detect_silence() expects int for silence_thresh parameter
            silence_ranges = detect_silence(
                audio,
                min_silence_len=400,
                silence_thresh=int(audio.dBFS - 14)  # ✅ FIX: Convert to int
            )

            # Step 6: Calculate silence duration and ratio
            silence_duration: int = sum((end - start) for start, end in silence_ranges)
            silence_ratio: float = silence_duration / len(audio) if len(audio) > 0 else 0.0

            # Step 7: Calculate speaking activity (inverse of silence)
            speaking_ratio: float = 1 - silence_ratio

            # Step 8: Calculate dynamic range (loudest - quietest)
            # Dynamic range indicates variation in volume
            dynamic_range: float = audio.max_dBFS - audio.dBFS if audio.dBFS != float("-inf") else 0.0
            
            # step 8.5: stress score in the range of 0-1
            stress_score = (energy * 2) + (silence_ratio * 0.8)
            stress_score = min(max(stress_score, 0), 1)

            #step 8.75: stress level 
            if stress_score < 0.3:
                stress_level = "low"
            elif stress_score < 0.6:
                stress_level = "medium"
            else:
                stress_level = "high"

            # Step 9: Return all extracted features as a dictionary
            return {
                "duration_sec": round(float(duration),4),
                "energy": round(float(energy),4),
                "zcr": round(float(zero_crossings),4),
                "silence_ratio": round(float(silence_ratio),4),
                "speaking_ratio": round(float(speaking_ratio),4),
                "dynamic_range": round(float(dynamic_range),2),
                "stress_score": round(float(stress_score), 3),
                "stress_level": stress_level
            }

        except Exception as e:
            # Log error for debugging  
            logger.error(f"[error] [VoiceAnalysisService] [features] {str(e)}")
            # Return empty dict on failure
            return {}

    # ================= EMOTION =================
    def _detect_emotion(self, f: Dict[str, Any]) -> Dict[str, Any]:
        """Detect emotion from speech features.
        
        TYPE ANNOTATIONS:
        - 'f: Dict[str, Any]': Parameter is a dictionary of features
        - '-> Dict[str, Any]': Returns a dictionary with emotion label
        
        Args:
            f (Dict[str, Any]): Features dictionary from _extract_speech_features()
                Keys: 'energy', 'silence_ratio', 'zcr', etc.
            
        Returns:
            Dict with emotion detection result:
            - 'label': Emotion classification as string
              * 'frustration': High energy + high ZCR (loud, tense speech)
              * 'hesitation': High silence ratio (many pauses, uncertain)
              * 'confidence': High energy + low silence (loud, continuous)
              * 'neutral': Default/other patterns
              
        EMOTION LOGIC :----
        Frustration: Speaker is aggressive or angry (loud, frequent consonants)
        Hesitation: Speaker is uncertain (lots of pauses and silence)
        Confidence: Speaker is assertive (loud voice, few pauses)
        Neutral: Everything else
        """
        # Extract features with safe defaults
        energy: float = f.get("energy", 0)
        silence: float = f.get("silence_ratio", 0)
        zcr: float = f.get("zcr", 0)

        # Step 1: Classify based on energy and ZCR
        if energy > 0.05 and zcr > 0.1:
            label: str = "frustration"  # Loud + lots of consonants
        # Step 2: Classify based on silence
        elif silence > 0.4:
            label: str = "hesitation"  # More than 40% silent
        # Step 3: Classify based on energy and silence
        elif energy > 0.03 and silence < 0.2:
            label: str = "confidence"  # Loud + few pauses
        # Step 4: Default classification
        else:
            label: str = "neutral"

        return {
            "label": label
        }

    # ================= COGNITIVE =================
    def _estimate_cognitive_state(self, f: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cognitive load from speech features.
        
        TYPE ANNOTATIONS:
        - 'f: Dict[str, Any]': Parameter is a dictionary of features
        - '-> Dict[str, Any]': Returns a dictionary with cognitive state
        
        Args:
            f (Dict[str, Any]): Features dictionary from _extract_speech_features()
                Keys: 'silence_ratio', 'dynamic_range', etc.
            
        Returns:
            Dict with cognitive state estimation:
            - 'cognitive_load': Classification as string
              * 'high': High cognitive load (lots of pauses while thinking)
              * 'medium': Moderate cognitive load
              * 'low': Low cognitive load (fluent, continuous speech)
            - 'speech_variability': Float value (dynamic_range) indicating
              how much the speaker varies volume/intensity
              
        COGNITIVE THEORY :---
        High cognitive load causes hesitations and pauses; speaker
        must think longer before responding. Measured by silence ratio.
        Speech variability (dynamic_range) also indicates effort.
        """
        # Extract features with safe defaults
        silence: float = f.get("silence_ratio", 0)
        dynamic: float = f.get("dynamic_range", 0)

        # Step 1: Classify cognitive load based on silence ratio
        if silence > 0.5:
            load: str = "high"  # More than 50% silent = thinking hard
        elif silence > 0.25:
            load: str = "medium"  # 25-50% silent = moderate thinking
        else:
            load: str = "low"  # Less than 25% silent = fluent speech

        return {
            "cognitive_load": load,
            "speech_variability": dynamic
        }
