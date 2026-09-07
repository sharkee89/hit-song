import os
import requests
import json
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
load_dotenv(os.path.join(project_root, '.env'))


def test_spotify_extended_api(artist_id="4tZwfgrHOc3mvqYlEYSvVi"):
    api_key = os.getenv('SPOTIFY_EXTENDED_AUDIO_FEATURES_API_KEY')
    api_host = os.getenv('SPOTIFY_EXTENDED_AUDIO_FEATURES_API_HOST')

    if not api_key:
        print("❌ Error: SPOTIFY_EXTENDED_AUDIO_FEATURES_API_KEY not found in .env file!")
        return

    url = f"https://{api_host}/v1/artists/{artist_id}"

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host,
        "Content-Type": "application/json"
    }

    print(f"🚀 Sending request for artist ID: {artist_id}...")

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("✅ Success! Data received.")
            data = response.json()
            print(json.dumps(data, indent=4))
        elif response.status_code == 401:
            print("❌ Error 401: API key not valid or expired.")
        elif response.status_code == 403:
            print("❌ Greška 403: unauthorized.")
        else:
            print(f"⚠️ Error. Status code: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Error in script: {e}")


if __name__ == "__main__":
    test_spotify_extended_api()