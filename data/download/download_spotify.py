import os
import requests
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

# 1. Učitavanje promenljivih iz .env fajla
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID_SPOTIFY")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_SPOTIFY")

# Provera da li su kredencijali uspešno povučeni
if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "Kredencijali nisu pronađeni! Proveri da li se u .env fajlu nalaze CLIENT_ID_SPOTIFY i CLIENT_SECRET_SPOTIFY."
    )

# 2. Inicijalizacija Spotipy klijenta
auth_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET
)
sp = spotipy.Spotify(auth_manager=auth_manager)

# 3. Primer ID-ja
track_id = "5iwz1NiezX7WWjnCgY5TH4"

try:
    # A) Preuzimanje metapodataka o pesmi (Za Context/Fusion Agenta)
    track_info = sp.track(track_id)

    track_name = track_info["name"]
    artist_name = track_info["artists"][0]["name"]
    popularity = track_info["popularity"]  # Target score (0-100)
    preview_url = track_info["preview_url"]  # Link do 30s MP3 audio fajla

    print(f"Pesma: {track_name} - {artist_name}")
    print(f"Popularnost: {popularity}")
    print(f"Preview URL: {preview_url}")

    # B) Preuzimanje 30s MP3 audio fajla (Za tvoj Audio Agent i Librosu)
    if preview_url:
        audio_data = requests.get(preview_url).content

        # Podesi direktorijum gde želiš da čuvaš audio fajlove
        os.makedirs("downloaded_audio", exist_ok=True)
        file_path = f"downloaded_audio/{track_id}.mp3"

        with open(file_path, "wb") as f:
            f.write(audio_data)

        print(f" Audio uspešno sačuvan na: {file_path}")
    else:
        print(" Ovaj track nema dostupan 30s preview na Spotify API-ju.")

except Exception as e:
    print(f"Greška prilikom skidanja: {e}")