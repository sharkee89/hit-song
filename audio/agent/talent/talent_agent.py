import ast
import json
import math
import os
import sys
import musicbrainzngs
import pandas as pd
import torch
from dotenv import load_dotenv

# Ensure root path resolution
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio.ai.ai_engine import AIEngine


class TalentAgent:

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TalentAgent] Execution Device: {self.device}")

        # Configure MusicBrainz client identity
        musicbrainzngs.set_useragent(
            app="AudioTalentAgent",
            version="1.0.0",
            contact="developer@example.com",
        )

        # Cache dataset dataframe to avoid reading CSV on every request
        self._dataset_df = None

    def _get_dataset(self) -> pd.DataFrame | None:
        """Lazy loader for the dataset CSV to optimize execution speed."""
        if self._dataset_df is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.abspath(
                os.path.join(current_dir, "..", "..", "..")
            )
            csv_path = os.path.join(
                base_dir,
                "data",
                "dataset",
                "SpotGenTrack",
                "Data_Sources",
                "spotify_artists.csv",
            )

            if os.path.exists(csv_path):
                try:
                    self._dataset_df = pd.read_csv(csv_path)
                except Exception as err:
                    print(f"❌ Error loading CSV dataset: {err}")
            else:
                print(f"⚠️ Dataset file not found at path: {csv_path}")

        return self._dataset_df

    def _fetch_musicbrainz_data(self, talent_name: str) -> dict:
        """Queries MusicBrainz API for official artist metadata and identifiers."""
        try:
            result = musicbrainzngs.search_artists(artist=talent_name, limit=1)
            artist_list = result.get("artist-list", [])

            if not artist_list:
                return {
                    "mbid": None,
                    "type": None,
                    "country": None,
                    "tags": [],
                    "found": False,
                }

            artist = artist_list[0]
            tags = [tag["name"] for tag in artist.get("tag-list", [])]

            return {
                "found": True,
                "mbid": artist.get("id"),
                "artist_type": artist.get("type"),
                "country": artist.get("country"),
                "tags": tags,
                "disambiguation": artist.get("disambiguation", ""),
            }
        except Exception as err:
            print(f"⚠️ MusicBrainz lookup failed: {err}")
            return {"found": False, "mbid": None, "error": str(err)}

    def _compute_market_features(
        self, followers: int, popularity: float
    ) -> dict:
        """
        Calculates mathematical market features (Log Followers, Stream Floor, Market Tier)
        for ML downstream tasks and model fusion.
        """
        # 1. Logarithmic Follower Scale
        log_followers = (
            round(math.log10(followers + 1), 3) if followers > 0 else 0.0
        )

        # 3. Projected Weekly Organic Stream Floor (15% conversion floor + popularity multiplier)
        conversion_rate = 0.15
        base_streams = followers * conversion_rate
        pop_multiplier = 1.0 + (popularity / 100.0)
        projected_floor = int(base_streams * pop_multiplier)

        return {
            "log_followers": log_followers,
            "projected_weekly_floor_streams": projected_floor,
            "fanbase_conversion_rate": conversion_rate,
        }

    def fetch_talent_data(self, talent_name: str) -> dict:
        """Retrieves combined talent metadata from SpotGenTrack dataset and MusicBrainz API."""
        df = self._get_dataset()
        spotify_data = {"found": False}

        # 1. Extract from local Spotify Dataset
        if df is not None:
            matches = df[df["name"].str.lower() == talent_name.lower()]

            if not matches.empty:
                artist_info = matches.iloc[0]
                tracks = (
                    matches[["track_id", "track_name_prev"]]
                    .dropna()
                    .drop_duplicates()
                    .to_dict(orient="records")
                )

                # Safely parse genres string into a list
                raw_genres = str(artist_info.get("genres", "[]"))
                try:
                    genres_list = ast.literal_eval(raw_genres)
                except (ValueError, SyntaxError):
                    genres_list = []

                followers = int(artist_info.get("followers", 0))
                popularity = float(artist_info.get("artist_popularity", 0.0))

                spotify_data = {
                    "found": True,
                    "spotify_id": str(artist_info.get("id", "")),
                    "artist_popularity": popularity,
                    "followers": followers,
                    "genres": genres_list,
                    "artist_type": str(artist_info.get("type", "")),
                    "tracks": tracks,
                    "computed_market_features": self._compute_market_features(
                        followers, popularity
                    ),
                }

        # 2. Query MusicBrainz API
        # mb_data = self._fetch_musicbrainz_data(talent_name)

        # 3. Consolidate Results
        return {
            "name": talent_name,
            "found": spotify_data.get("found", False),
            "spotify_data": spotify_data,
            # "musicbrainz_data": mb_data,
        }

    def get_ai_analysis(self, talent_data: dict) -> str:
        """Passes aggregated talent metadata to AIEngine for market profile report."""
        ai_engine = AIEngine()
        # Fallback if specific talent method isn't implemented on AIEngine yet
        if hasattr(ai_engine, "get_talent_detail_analysis"):
            return ai_engine.get_talent_detail_analysis(talent_data)
        return "AI analysis engine pending integration."

    def get_feature_vector(self, talent_data: dict) -> torch.Tensor:
        """Extracts a normalized, flat numerical tensor for ML model fusion."""
        if not talent_data.get("found", False):
            # Fallback vector for unknown artists
            return torch.tensor([0.0, 0.0, 0.0, 0.15], dtype=torch.float32)

        spotify_data = talent_data.get("spotify_data", {})
        market_features = spotify_data.get("computed_market_features", {})

        popularity_norm = spotify_data.get("artist_popularity", 0.0) / 100.0
        log_followers = market_features.get("log_followers", 0.0)
        # Log scaling for projected streams to match network weights
        log_projected_streams = math.log10(
            market_features.get("projected_weekly_floor_streams", 0) + 1
        )
        conversion_rate = market_features.get("fanbase_conversion_rate", 0.15)

        return torch.tensor(
            [
                popularity_norm,
                log_followers,
                log_projected_streams,
                conversion_rate,
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def process(self, talent_name: str) -> str:
        """Full end-to-end execution pipeline for TalentAgent."""
        talent_data = self.fetch_talent_data(talent_name)
        ai_analysis = self.get_ai_analysis(talent_data)
        talent_data["ai_analysis"] = ai_analysis

        return json.dumps(talent_data, indent=4)


if __name__ == "__main__":
    load_dotenv()
    talent_name = os.getenv("TALENT_NAME", "The Weeknd")
    agent = TalentAgent()

    if talent_name:
        print(
            f"🚀 Starting full TalentAgent pipeline processing for: {talent_name}..."
        )
        data = json.loads(agent.process(talent_name))
        print(data)
        print(agent.get_feature_vector(data))
        print("✅ TalentAgent executed successfully!")
    else:
        print("❌ TALENT_NAME isn't defined in .env.")