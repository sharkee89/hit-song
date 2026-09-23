import anthropic

class AIEngine:

    def __init__(self):
        self.agent = "claude"
        self.client = anthropic.Anthropic()


    def get_audio_detail_analysis(self, audio_data: str):
        # response = self.client.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     max_tokens=2000,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": f"Izvrši detaljnu muzičku i tehničku analizu sledećih audio metrika:\n\n{audio_data}"
        #         }
        #     ]
        # )
        # print("\n🤖 Claude analiza:\n" + response.content[0].text)
        # return response.content[0].text
        return "Ovo je lažna analiza audio fajla za potrebe testiranja koda."