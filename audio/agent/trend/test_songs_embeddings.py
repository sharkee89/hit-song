import os
import json
import logging
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_distances

from audio.agent.audio_agent import AudioAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

load_dotenv()

AUDIO_FILES = {
    "SAME_SONG_DIFFERENT_VERSION_1": os.getenv(
        "SAME_SONG_DIFFERENT_VERSION_1"
    ),
    "SAME_SONG_DIFFERENT_VERSION_2": os.getenv(
        "SAME_SONG_DIFFERENT_VERSION_2"
    ),
    "SAME_SONG_DIFFERENT_VERSION_3": os.getenv(
        "SAME_SONG_DIFFERENT_VERSION_3"
    ),
}

OUTPUT_DIR = Path("logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "audio_embedding_similarity.json"


def validate_audio_files():
  for name, path in AUDIO_FILES.items():
    if not path:
      raise ValueError(f"Nedostaje putanja u .env fajlu za: {name}")

    if not Path(path).exists():
      raise FileNotFoundError(f"Audio fajl ne postoji: {path}")


def extract_embedding(audio_agent, audio_path):
  # Korišćenje ispravne metode iz AudioAgent klase
  embedding = audio_agent.extract_features(audio_path)

  if embedding is None:
    raise ValueError(f"Embedding nije generisan za fajl: {audio_path}")

  if hasattr(embedding, "detach"):
    embedding = embedding.detach().cpu().numpy()

  embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

  if embedding.size == 0:
    raise ValueError(f"Embedding je prazan za fajl: {audio_path}")

  return embedding


def main():
  logging.info("=" * 90)
  logging.info("TEST: AUDIO EMBEDDING SIMILARITY BETWEEN DIFFERENT VERSIONS")
  logging.info("=" * 90)

  validate_audio_files()

  audio_agent = AudioAgent()

  embeddings = {}
  metadata = {}

  for name, audio_path in AUDIO_FILES.items():
    logging.info("-" * 90)
    logging.info("Generisanje embedding-a za: %s", audio_path)

    embedding = extract_embedding(audio_agent, audio_path)

    embeddings[name] = embedding
    metadata[name] = {
        "audio_file": audio_path,
        "embedding_dimension": int(embedding.shape[0]),
    }

    logging.info("Embedding generisan: %s dimenzija", embedding.shape[0])

  names = list(embeddings.keys())
  matrix = np.vstack([embeddings[name] for name in names])

  distance_matrix = cosine_distances(matrix)

  logging.info("")
  logging.info("=" * 90)
  logging.info("COSINE DISTANCE MATRICA")
  logging.info("=" * 90)

  logging.info("%-32s%s", "", "".join(f"{name:>32}" for name in names))

  for index, name in enumerate(names):
    row = "".join(
        f"{distance_matrix[index][column]:32.6f}"
        for column in range(len(names))
    )
    logging.info("%-32s%s", name, row)

  pair_results = []

  logging.info("")
  logging.info("=" * 90)
  logging.info("POREĐENJE VERZIJA")
  logging.info("=" * 90)

  for i in range(len(names)):
    for j in range(i + 1, len(names)):
      name_a = names[i]
      name_b = names[j]
      distance = float(distance_matrix[i][j])

      result = {
          "song_a": name_a,
          "song_b": name_b,
          "audio_file_a": AUDIO_FILES[name_a],
          "audio_file_b": AUDIO_FILES[name_b],
          "cosine_distance": distance,
      }

      pair_results.append(result)

      logging.info("%s VS %s | cosine_distance = %.6f", name_a, name_b, distance)

  results = {
      "test_name": "audio_embedding_similarity_between_versions",
      "songs": metadata,
      "cosine_distance_matrix": {
          names[i]: {
              names[j]: float(distance_matrix[i][j]) for j in range(len(names))
          }
          for i in range(len(names))
      },
      "pairwise_comparisons": pair_results,
  }

  with open(RESULTS_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

  logging.info("")
  logging.info("=" * 90)
  logging.info("TEST ZAVRŠEN")
  logging.info("JSON rezultati: %s", RESULTS_FILE)
  logging.info("=" * 90)


if __name__ == "__main__":
  main()