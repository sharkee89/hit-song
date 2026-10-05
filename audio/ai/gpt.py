import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class Gpt:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_analysis(self, prompt: str, audio_file_path: str = None) -> str:
        try:
            audio_transcript = ""

            # Korak 1: Ako postoji audio fajl, prvo ga transkribujemo preko Whisper-a
            if audio_file_path and os.path.exists(audio_file_path):
                with open(audio_file_path, "rb") as audio_file:
                    transcript_response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    audio_transcript = getattr(transcript_response, "text", str(transcript_response))

            # Korak 2: Formiramo kompletan prompt koji uključuje i originalni prompt i transkript pesme/audio zapisa
            full_prompt = prompt
            if audio_transcript:
                full_prompt += f"\n\n[Audio Transcription / Lyrical & Acoustic Content Analysis]:\n{audio_transcript}"

            # Korak 3: Šaljemo tekstualni zahtev GPT-u
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Može i gpt-4o
                messages=[  # type: ignore
                    {
                        "role": "system",
                        "content": "You are an expert in the music industry and hit prediction."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0.7,
            )

            analysis_text = response.choices[0].message.content.strip()
            return analysis_text

        except Exception as e:
            return f"Error communicating with OpenAI API: {e}"