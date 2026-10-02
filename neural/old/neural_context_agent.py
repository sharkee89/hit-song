import os
import musicbrainzngs
import json
import requests
import math
from dotenv import load_dotenv

load_dotenv()

class NeuralContextAgent:
    def __init__(self, name="Neural: ContextAnalyst"):
        # MusicBrainz Setup
        musicbrainzngs.set_useragent(
            os.getenv("MUSICBRAINZ_APP", "NeuralHitPredictor"),
            "1.0",
            os.getenv("STUDENT_EMAIL", "example@student.com")
        )
        self.name = name
        self.rapidapi_key = os.getenv('SPOTIFY_EXTENDED_AUDIO_FEATURES_API_KEY')
        self.rapidapi_host = os.getenv('SPOTIFY_EXTENDED_AUDIO_FEATURES_API_HOST')

    def _get_spotify_data(self, artist_name):
        """Pretražuje Spotify API za dodatne metrike o popularnosti."""
        url = f"https://{self.rapidapi_host}/v1/search"
        querystring = {"q": artist_name, "type": "artist", "limit": "1"}
        headers = {
            "x-rapidapi-key": self.rapidapi_key,
            "x-rapidapi-host": self.rapidapi_host
        }
        try:
            response = requests.get(url, headers=headers, params=querystring)
            # POPRAVLJENO: status_code umesto status_status
            if response.status_code == 200:
                data = response.json()
                artists = data.get('artists', {}).get('items', [])
                return artists[0] if artists else None
        except Exception as e:
            print(f"[{self.name}] Spotify API Error: {e}")
        return None

    def _calculate_hybrid_signal(self, mb_data, sp_data):
        """Kombinuje MusicBrainz (istorijski autoritet) i Spotify (tržišna snaga)."""
        # 1. MusicBrainz Score (Max 0.3) - fokus na bazu i poreklo
        mb_score = 0.05
        tags = mb_data.get('tag-list', [])
        mb_score += min(len(tags) * 0.02, 0.15)
        if mb_data.get('area'): mb_score += 0.05
        if mb_data.get('type') in ['Group', 'Orchestra']: mb_score += 0.05

        # 2. Spotify Popularity Score (Max 0.5) - direktna tržišna snaga
        sp_pop_score = 0.0
        if sp_data:
            popularity = sp_data.get('popularity', 0)
            sp_pop_score = (popularity / 100) * 0.5

        # 3. Spotify Followers Impact (Max 0.2) - logaritamski uticaj fan baze
        sp_follow_score = 0.0
        if sp_data:
            followers = sp_data.get('followers', {}).get('total', 0)
            if followers > 0:
                # Normalizacija na bazi logaritamske skale (do 100M pratilaca)
                sp_follow_score = min((math.log10(followers) / 8) * 0.2, 0.2)

        total_activation = mb_score + sp_pop_score + sp_follow_score
        return round(min(total_activation, 1.0), 4)

    def get_artist_report(self, artist_name):
        print(f"[{self.name}]: Fetching hybrid neural context for: {artist_name}...")
        try:
            if not artist_name or artist_name.lower() == "unknown":
                return self._format_report({"name": "Unknown"}, None, 0.001)

            mb_result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
            sp_data = self._get_spotify_data(artist_name)

            if not mb_result['artist-list'] and not sp_data:
                return self._format_report({"name": artist_name}, None, 0.005)

            mb_artist = mb_result['artist-list'][0] if mb_result['artist-list'] else {}
            activation_value = self._calculate_hybrid_signal(mb_artist, sp_data)

            return self._format_report(mb_artist, sp_data, activation_value)

        except Exception as e:
            print(f"[{self.name}] General error: {str(e)}")
            return {"status": "error", "activation_value": 0.1}

    def _format_report(self, mb_artist, sp_data, activation_value):
        """Formatira finalni izveštaj sa hibridnim podacima."""
        return {
            "agent_name": self.name,
            "activation_value": activation_value,
            "status": "completed",
            "findings": {
                "artist_info": {
                    "name": sp_data.get('name') if sp_data else mb_artist.get('name'),
                    "type": mb_artist.get('type', 'Person/Unknown'),
                    "origin": mb_artist.get('area', {}).get('name', 'Unknown'),
                    "spotify_popularity": sp_data.get('popularity') if sp_data else 0,
                    "followers": sp_data.get('followers', {}).get('total') if sp_data else 0
                },
                "authority_metrics": {
                    "raw_signal": activation_value,
                    "market_strength": "High" if activation_value > 0.7 else "Medium" if activation_value > 0.4 else "Low"
                }
            }
        }

if __name__ == "__main__":
    agent = NeuralContextAgent()
    # Testiramo na Taylor Swift koja je ranije imala 0.7, sada bi trebala biti blizu 1.0
    report = agent.get_artist_report("Taylor Swift")
    print(json.dumps(report, indent=2))
    report = agent.get_artist_report("Weeknd")
    print(json.dumps(report, indent=2))
    report = agent.get_artist_report("The Beatles")
    print(json.dumps(report, indent=2))
