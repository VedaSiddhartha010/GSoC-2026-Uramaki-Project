"""
Speech Emotion Model
Uses HuggingFace audio classification pipeline
"""

import torch
import torch.nn as nn
from transformers import pipeline
from typing import Dict , Any , List


class SpeechEmotionModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super(SpeechEmotionModel, self).__init__()

        self.debug = config.get('debug')

        emotion_config = config.get('speech_emotion')
        if not emotion_config:
            raise ValueError("'speech_emotion' not found in config")

        self.config = emotion_config.get('default')
        if not self.config:
            raise ValueError("'default' speech_emotion config missing")

        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        print(f"Loading Speech Emotion Model: {self.model_name}")

        self.pipeline = pipeline(
            task="audio-classification",
            model=self.model_name,
            device=self.device
        )

    def forward(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform emotion classification
        """
        outputs = self.pipeline(audio_path)

        # Safe handling
        if not isinstance(outputs, list) or len(outputs) == 0:
            return {
                "emotion": {
                    "label": "unknown",
                    "score": 0.0
                }
            }

        top = outputs[0]

        return {
            "emotion": {
                "label": top.get("label", "unknown"),
                "score": float(top.get("score", 0.0))
            }
        }