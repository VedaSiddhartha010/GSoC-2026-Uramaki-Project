import yaml
from app.services.voice_analysis_service import VoiceAnalysisService

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

service = VoiceAnalysisService(config)

result = service.analyze("./samples/sample_1.mp3")

print(result)