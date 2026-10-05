import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class Gpt:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_analysis(self, prompt: str) -> str:
        return("gpt example analysis")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[  # type: ignore
                    {
                        "role": "system",
                        "content": "You are an expert in the music industry and hit prediction."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
            )

            analysis_text = response.choices[0].message.content.strip()

            return analysis_text

        except Exception as e:
            return f"Error communicating with OpenAI API: {e}"