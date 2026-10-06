import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Gpt:

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def get_analysis(self, prompt: str, audio_file_path: str = None) -> str:
        try:

            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            if audio_file_path and os.path.exists(audio_file_path):

                with open(audio_file_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()

                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "mp3"
                        }
                    }
                )

            response = self.client.chat.completions.create(
                model="gpt-audio",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in the music industry "
                            "and hit prediction. Analyze the provided "
                            "audio directly, including musical and "
                            "production characteristics."
                        )
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error communicating with OpenAI API: {e}"