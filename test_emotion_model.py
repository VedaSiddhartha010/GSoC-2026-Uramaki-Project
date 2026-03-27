from app.models.speech_emotion_model import SpeechEmotionModel

config = {
    "debug": True,
    "speech_emotion": {
        "default": {
            "model_name": "superb/wav2vec2-base-superb-er",
            "device": -1
        }
    }
}

model = SpeechEmotionModel(config)

result = model("./samples/sample_1.mp3")
print("Label:", result["emotion"]["label"])
print("Score:", result["emotion"]["score"])