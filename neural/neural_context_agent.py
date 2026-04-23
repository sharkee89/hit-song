import os
import musicbrainzngs
import json
from dotenv import load_dotenv
load_dotenv()

class NeuralContextAgent:
    def __init__(self, name="Neural: ContextAnalyst"):
        musicbrainzngs.set_useragent(os.getenv("MUSICBRAINZ_APP"), "1.0", os.getenv("STUDENT_EMAIL"))
        self.name = name

    def _calculate_authority_signal(self, artist_data):
        score = 0.1

        tags = artist_data.get('tag-list', [])
        tag_count = len(tags)
        score += min(tag_count * 0.05, 0.5)

        if artist_data.get('type') in ['Group', 'Orchestra']:
            score += 0.1

        if artist_data.get('area'):
            score += 0.1

        return round(min(score, 1.0), 4)

    def get_artist_report(self, artist_name):
        print(f"[{self.name}]: Fetching neural context for: {artist_name}...")
        try:
            if not artist_name or artist_name.lower() == "unknown":
                return self._format_report({"name": "Unknown"}, 0.001)

            result = musicbrainzngs.search_artists(artist=artist_name, limit=1)

            if not result['artist-list']:
                return self._format_report({"name": artist_name}, 0.005)

            artist = result['artist-list'][0]
            activation_value = self._calculate_authority_signal(artist)

            return self._format_report(artist, activation_value)

        except Exception as e:
            print(f"[{self.name}] Error during retrieval: {str(e)}")
            return {"status": "error", "activation_value": 0.1}

    def _format_report(self, artist, activation_value):
        return {
            "agent_name": self.name,
            "activation_value": activation_value,
            "status": "completed",
            "findings": {
                "artist_info": {
                    "name": artist.get('name'),
                    "type": artist.get('type', 'Unknown'),
                    "origin": artist.get('area', {}).get('name', 'Unknown'),
                    "tags": [tag['name'] for tag in artist.get('tag-list', [])[:5]] if 'tag-list' in artist else []
                },
                "authority_metrics": {
                    "raw_signal": activation_value,
                    "market_strength": "High" if activation_value > 0.7 else "Medium" if activation_value > 0.3 else "Low"
                }
            }
        }


if __name__ == "__main__":
    agent = NeuralContextAgent()
    print(json.dumps(agent.get_artist_report("Taylor Swift"), indent=2))