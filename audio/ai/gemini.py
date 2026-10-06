import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class Gemini:

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def get_analysis(self, prompt: str, audio_file_path: str = None) -> str:
        try:
            contents = [prompt]
            uploaded_file = None

            # Ispravljeno: os.path.exists umesto os.path-exists
            if audio_file_path and os.path.exists(audio_file_path):
                uploaded_file = self.client.files.upload(file=audio_file_path)
                contents.append(uploaded_file)

            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert in the music industry and hit prediction.",
                    temperature=0.7,
                )
            )

            if uploaded_file:
                self.client.files.delete(name=uploaded_file.name)

            return response.text.strip()

        except Exception as e:
            return f"Error communicating with Gemini API: {e}"