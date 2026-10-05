import os
import requests
from dotenv import load_dotenv

load_dotenv()


class Deepseek:

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/chat/completions"

    def get_analysis(self, geminiAnalysis: str, gptAnalysis: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Formiranje detaljnog prompta sa unetim analizama oba modela
        user_prompt = (
            "Do the validation and comparison between these two AI outputs (Gemini and GPT) for a song analysis. "
            "Find their agreement and disagreement points, and synthesize a final consensus report.\n\n"
            f"--- GEMINI ANALYSIS ---\n{geminiAnalysis}\n\n"
            f"--- GPT ANALYSIS ---\n{gptAnalysis}\n\n"
            "Provide the final output strictly in a JSON response format containing the following properties: "
            "production_quality, audio_quality, prediction_detail_analysis, possible_improvements, prediction_grade.\n"
            "For every JSON property, provide two fields: 'numeric_value' (a number from 0 to 100) and 'analysis' (a descriptive text synthesizing both models). "
            "Note: The value for 'possible_improvements' should be higher if the demand for improvements is lower and vice versa."
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert music industry arbitrator and hit prediction meta-analyst."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.7,
            "stream": False
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            return f"Error communicating with DeepSeek API: {e}"