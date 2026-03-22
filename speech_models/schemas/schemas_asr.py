from dataclasses import dataclass


@dataclass
class ASRSchema:
    key: str
    audio_path: str
    transcription: str
    duration_seconds: float
