import pandas as pd
import os
import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return " ".join(text.split())


def run_matching():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data', 'dataset')
    top_songs_path = os.path.join(data_dir, 'trend_2.csv')
    enriched_path = os.path.join(data_dir, 'dataset_enriched.csv')
    output_path = os.path.join(data_dir, 'dataset_trend.csv.csv')

    if not os.path.exists(top_songs_path) or not os.path.exists(enriched_path):
        print(f"❌ Error: check if files are in {data_dir}")
        return

    print("🚀 Loading files...")
    df_top = pd.read_csv(top_songs_path)
    df_enriched = pd.read_csv(enriched_path)
    print("🧹 Cleaning data for parring precision...")

    # From 'The Weeknd - Blinding Lights' getting only 'Blinding Lights'
    df_top['match_name'] = df_top['Songs & Artist'].apply(
        lambda x: clean_text(x.split('-')[-1] if '-' in str(x) else x)
    )
    df_enriched['match_name'] = df_enriched['track_name'].apply(clean_text)
    df_enriched = df_enriched.sort_values('popularity', ascending=False)
    df_enriched = df_enriched.drop_duplicates(subset=['match_name'], keep='first')
    print("🔗 Connecting tables...")
    final_df = pd.merge(df_top, df_enriched, on='match_name', how='left')

    cols_to_drop = ['match_name']
    if 'Unnamed: 0' in final_df.columns:
        cols_to_drop.append('Unnamed: 0')

    final_df = final_df.drop(columns=cols_to_drop)
    total_songs = len(df_top)
    matched_songs = final_df['track_id'].notna().sum()

    print("-" * 30)
    print(f"✅ Done!")
    print(f"📊 Total songs in the list: {total_songs}")
    print(f"🎯 Successfully cross matched with audio data: {matched_songs}")
    print(f"❌ Not found in database: {total_songs - matched_songs}")
    print("-" * 30)

    final_df.to_csv(output_path, index=False)
    print(f"💾 Result saved in: {output_path}")


if __name__ == "__main__":
    run_matching()