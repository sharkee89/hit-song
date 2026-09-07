import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from audio_quality.service.audio_quality_service import AudioQualityService

if __name__ == "__main__":
    try:
        service = AudioQualityService()
        data = service.get_data()
        print(data)
    except (ValueError, RuntimeError) as err:
        print(f"Error: {err}")
