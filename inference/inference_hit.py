import json
import os
from dotenv import load_dotenv
from audio.ai.deepseek import Deepseek
from audio.ai.gemini import Gemini
from audio.ai.gpt import Gpt
from audio.utils.audio_utils import get_librosa_data

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_ai_analysis(features_str: str) -> str:
    prompt = (
        f"You are a top music producer, analyst, and A&R expert for Spotify. "
        f"Analyze the potential of the song but based on audio features to become a hit based on the following information:\n"
        f"{features_str}\n\n"
        f"Provide a detail, professional assessment of the commercial potential, genre fit, "
        f"and a recommendation for market placement."
        f"Also provide advices for edit song in audio way to make it more appeal to today's market need."
    )
    print(prompt)
    deep_seek_analysis =  Deepseek().get_analysis(prompt)
    gemini_analysis = Gemini().get_analysis(prompt)
    gpt_analysis = Gpt().get_analysis(prompt)
    return {
        "deep_seek": deep_seek_analysis,
        "gemini": gemini_analysis,
        "gpt": gpt_analysis
    }

def get_gpt_analysis(prompt: str) -> str:
    return "gpt example analysis"

def main():
    print("Executing")
    audio_data = get_librosa_data()
    ai_analysis = get_ai_analysis(str(audio_data))
    data = {
        "value": audio_data,
        "ai": ai_analysis
    }
    print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()