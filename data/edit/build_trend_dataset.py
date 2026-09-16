from pathlib import Path
import pandas as pd
import re

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATASET_DIR = CURRENT_DIR.parent / "dataset"

path_trend_1 = DATASET_DIR / "trend_1.csv"
path_trend_2 = DATASET_DIR / "trend_2.csv"
path_trend_3 = DATASET_DIR / "trend_3.csv"

SPOTIFY_PARQUET_PATH = (
        PROJECT_ROOT
        / "data"
        / "dataset"
        / "spotify_tracks_with_artists.parquet"
)


def clean_streams(value):
    if pd.isna(value):
        return 0
    cleaned = re.sub(r'[^0-9]', '', str(value))
    try:
        return int(cleaned) if cleaned else 0
    except ValueError:
        return 0


def normalize_string(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def process_datasets():
    all_records = []

    # 1. Obrada trend_1.csv
    if path_trend_1.exists():
        df1 = pd.read_csv(path_trend_1)
        for _, row in df1.iterrows():
            song = str(row.get('track_name', '')).strip()
            artist = str(row.get('artist_name', '')).strip()
            streams = clean_streams(row.get('streams', 0))

            if song and artist:
                all_records.append({
                    "Songs & Artist": f"{artist} - {song}",
                    "Streams": streams
                })

    # 2. Obrada trend_2.csv
    if path_trend_2.exists():
        df2 = pd.read_csv(path_trend_2)
        for _, row in df2.iterrows():
            song_artist = str(row.get('Songs & Artist', '')).strip()
            streams = clean_streams(row.get('Streams', 0))

            if song_artist:
                all_records.append({
                    "Songs & Artist": song_artist,
                    "Streams": streams
                })

    # 3. Obrada trend_3.csv (YouTube)
    if path_trend_3.exists():
        df3 = pd.read_csv(path_trend_3)
        for _, row in df3.iterrows():
            title = str(row.get('title', '')).strip()
            channel = str(row.get('channel', '')).strip()
            view_count = clean_streams(row.get('view_count', 0))

            if title and channel:
                all_records.append({
                    "Songs & Artist": f"{channel} - {title}",
                    "Streams": view_count
                })

    # Deduplikacija (zadržavanje unosa sa najvećim brojem strimova)
    unique_songs = {}
    for record in all_records:
        raw_key = record["Songs & Artist"]
        norm_key = normalize_string(raw_key)
        streams = record["Streams"]

        if norm_key in unique_songs:
            if streams > unique_songs[norm_key]["Streams"]:
                unique_songs[norm_key] = {
                    "Songs & Artist": raw_key,
                    "Streams": streams
                }
        else:
            unique_songs[norm_key] = {
                "Songs & Artist": raw_key,
                "Streams": streams
            }

    # Priprema finalne liste i sortiranje po strimovima opadajuće
    final_rows = list(unique_songs.values())
    final_df = pd.DataFrame(final_rows)

    if not final_df.empty:
        final_df = final_df.sort_values(by="Streams", ascending=False).reset_index(drop=True)

        # Distribucija poena od 1.0 do 0.5 ravnomerno
        n = len(final_df)
        if n > 1:
            final_df['points'] = [1.0 - (i / (n - 1)) * 0.5 for i in range(n)]
        else:
            final_df['points'] = [1.0]

    output_path = DATASET_DIR / "trends.csv"
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Uspešno generisan i sačuvan: {output_path}")

    return final_df


def match_and_align_datasets(df_trends):
    if not SPOTIFY_PARQUET_PATH.exists():
        print(f"❌ Fajl {SPOTIFY_PARQUET_PATH} ne postoji na zadatoj putanji.")
        return

    df_spotify = pd.read_parquet(SPOTIFY_PARQUET_PATH)
    print(f"📂 Učitan spotify_tracks_with_artists.parquet sa {len(df_spotify)} redova.")

    # Kreiranje rečnika parova (izvođač, pesma) -> points
    trends_set = {}
    for _, row in df_trends.iterrows():
        song_artist = str(row.get("Songs & Artist", ""))
        points = row.get("points", 0.0)

        if " - " in song_artist:
            parts = song_artist.split(" - ", 1)
            artist_part = normalize_string(parts[0])
            song_part = normalize_string(parts[1])
            if artist_part and song_part:
                trends_set[(artist_part, song_part)] = points

    df_spotify["trend_alignment"] = 0.0
    found_count = 0

    # Prolazimo kroz redove gde imamo kolone 'name' (pesma) i 'artist_name' (izvođač iz spojenog fajla)
    for idx, row in df_spotify.iterrows():
        track_name = row.get("name", "")
        artist_name = row.get("artist_name", "")

        norm_track = normalize_string(track_name)
        norm_artist = normalize_string(artist_name)

        if norm_artist and norm_track:
            pair_key = (norm_artist, norm_track)
            if pair_key in trends_set:
                df_spotify.at[idx, "trend_alignment"] = trends_set[pair_key]
                found_count += 1

    output_spotify_path = SPOTIFY_PARQUET_PATH.parent / "spotify_tracks_with_artists_aligned.parquet"
    df_spotify.to_parquet(output_spotify_path, index=False)

    print(f"✅ Poređenje i usklađivanje završeno!")
    print(f"🔍 Ukupno pronađenih i povezanih redova: {found_count} od {len(df_spotify)} ukupnih.")
    print(f"💾 Sačuvano u novi parquet fajl: {output_spotify_path}")


if __name__ == "__main__":
    df_trends = process_datasets()
    if not df_trends.empty:
        match_and_align_datasets(df_trends)