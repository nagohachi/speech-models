from dataclasses import dataclass


@dataclass
class TTSSchema:
    key: str
    audio_path: str
    phonemes: str
    duration_seconds: float
