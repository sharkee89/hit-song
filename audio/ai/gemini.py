import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class Gemini:

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def get_analysis(self, prompt: str) -> str:
        return "gemini example analysis"
        try:
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert in the music industry and hit prediction.",
                    temperature=0.7,
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Error communicating with Gemini API: {e}"