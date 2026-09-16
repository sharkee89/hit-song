# test_songs_trend_alignment.py

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from itertools import combinations

from dotenv import load_dotenv

from audio.agent.trend.trend_agent import train_and_predict_trend


# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------

load_dotenv()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = LOG_DIR / f"same_songs_different_version_{timestamp}.log"
JSON_FILE = LOG_DIR / f"same_songs_different_version_{timestamp}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ENV_VARIABLES = [
    "SAME_SONG_DIFFERENT_VERSION_1",
    "SAME_SONG_DIFFERENT_VERSION_2",
    "SAME_SONG_DIFFERENT_VERSION_3",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_audio_paths():
    """
    Učitava tri audio fajla iz .env fajla i proverava njihove putanje.
    """
    audio_paths = {}

    for variable_name in ENV_VARIABLES:
        value = os.getenv(variable_name)

        if not value:
            raise ValueError(
                f"Promenljiva '{variable_name}' nije definisana u .env fajlu."
            )

        audio_path = Path(value).expanduser().resolve()

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Fajl za '{variable_name}' ne postoji: {audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(
                f"Putanja za '{variable_name}' nije fajl: {audio_path}"
            )

        audio_paths[variable_name] = audio_path

    return audio_paths


def make_json_serializable(value):
    """
    Pretvara rezultate TrendAgent-a u JSON-kompatibilan oblik.
    """
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return value


def extract_summary(result):
    """
    Izvlači ključne vrednosti iz rezultata funkcije
    train_and_predict_trend().
    """
    if not result:
        return {
            "trend_alignment": None,
            "cosine_distance": None,
            "trend_feature_tensor": None,
        }

    return {
        "trend_alignment": result.get("trend_alignment"),
        "cosine_distance": result.get("cosine_distance"),
        "trend_feature_tensor": make_json_serializable(
            result.get("trend_feature_tensor")
        ),
    }


def absolute_difference(value_1, value_2):
    """
    Računa apsolutnu razliku između dve numeričke vrednosti.
    """
    if value_1 is None or value_2 is None:
        return None

    return abs(float(value_1) - float(value_2))


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

def main():
    logger.info("=" * 90)
    logger.info("TEST: SAME SONG - DIFFERENT VERSIONS")
    logger.info("=" * 90)

    audio_paths = get_audio_paths()

    logger.info("Audio fajlovi iz .env fajla:")

    for variable_name, audio_path in audio_paths.items():
        logger.info("%s = %s", variable_name, audio_path)

    results = {}

    # -----------------------------------------------------
    # Analiza svakog fajla preko TrendAgent pipeline-a
    # -----------------------------------------------------

    for variable_name, audio_path in audio_paths.items():
        logger.info("")
        logger.info("-" * 90)
        logger.info("Pokretanje TrendAgent analize: %s", variable_name)
        logger.info("Audio fajl: %s", audio_path)
        logger.info("-" * 90)

        try:
            # TrendAgent u dostavljenom kodu nema klasu niti analyze() metodu.
            # Njegova glavna funkcija za predikciju je train_and_predict_trend().
            #
            # Funkcija očekuje:
            #   1. df_filtered
            #   2. target_audio_path
            #
            # Zato se dataset i embedding cache pripremaju u nastavku.
            from trend_agent import (
                PARQUET_PATH,
                EMBEDDINGS_DIR,
                extract_and_cache_embeddings,
            )
            import pandas as pd

            if not PARQUET_PATH.exists():
                raise FileNotFoundError(
                    f"Parquet dataset nije pronađen: {PARQUET_PATH}"
                )

            logger.info("Učitavanje parquet dataseta...")
            df = pd.read_parquet(PARQUET_PATH)

            df_filtered = df[df["trend_alignment"] >= 0.5].copy()

            logger.info(
                "Broj filtriranih trend pesama: %d",
                len(df_filtered),
            )

            logger.info("Provera/generisanje embedding cache-a...")
            extract_and_cache_embeddings(df_filtered)

            logger.info("Pokretanje train_and_predict_trend()...")

            raw_result = train_and_predict_trend(
                df_filtered=df_filtered,
                target_audio_path=str(audio_path),
            )

            summary = extract_summary(raw_result)

            results[variable_name] = {
                "audio_path": str(audio_path),
                "trend_alignment": summary["trend_alignment"],
                "cosine_distance": summary["cosine_distance"],
                "trend_feature_tensor": summary["trend_feature_tensor"],
                "raw_result": make_json_serializable(raw_result),
            }

            logger.info(
                "Rezultat za %s:",
                variable_name,
            )
            logger.info(
                "trend_alignment = %s",
                summary["trend_alignment"],
            )
            logger.info(
                "cosine_distance = %s",
                summary["cosine_distance"],
            )
            logger.info(
                "trend_feature_tensor = %s",
                summary["trend_feature_tensor"],
            )

        except Exception as exc:
            logger.exception(
                "Greška pri obradi fajla '%s': %s",
                audio_path,
                exc,
            )

            results[variable_name] = {
                "audio_path": str(audio_path),
                "error": str(exc),
            }

    # -----------------------------------------------------
    # Poređenje svih parova
    # -----------------------------------------------------

    logger.info("")
    logger.info("=" * 90)
    logger.info("POREĐENJE SVIH VERZIJA")
    logger.info("=" * 90)

    successful_results = {
        name: result
        for name, result in results.items()
        if "error" not in result
    }

    for name_1, name_2 in combinations(successful_results.keys(), 2):
        result_1 = successful_results[name_1]
        result_2 = successful_results[name_2]

        trend_difference = absolute_difference(
            result_1.get("trend_alignment"),
            result_2.get("trend_alignment"),
        )

        distance_difference = absolute_difference(
            result_1.get("cosine_distance"),
            result_2.get("cosine_distance"),
        )

        logger.info("")
        logger.info("%s VS %s", name_1, name_2)
        logger.info(
            "Razlika trend_alignment: %s",
            trend_difference,
        )
        logger.info(
            "Razlika cosine_distance: %s",
            distance_difference,
        )

    # -----------------------------------------------------
    # Konzistentnost
    # -----------------------------------------------------

    logger.info("")
    logger.info("=" * 90)
    logger.info("KONZISTENTNOST REZULTATA")
    logger.info("=" * 90)

    trend_values = [
        result["trend_alignment"]
        for result in successful_results.values()
        if result.get("trend_alignment") is not None
    ]

    distance_values = [
        result["cosine_distance"]
        for result in successful_results.values()
        if result.get("cosine_distance") is not None
    ]

    if trend_values:
        trend_range = max(trend_values) - min(trend_values)

        logger.info("trend_alignment vrednosti: %s", trend_values)
        logger.info("Raspon trend_alignment vrednosti: %.6f", trend_range)

    if distance_values:
        distance_range = max(distance_values) - min(distance_values)

        logger.info("cosine_distance vrednosti: %s", distance_values)
        logger.info("Raspon cosine_distance vrednosti: %.6f", distance_range)

    # -----------------------------------------------------
    # Čuvanje JSON rezultata
    # -----------------------------------------------------

    with JSON_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(results),
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("")
    logger.info("=" * 90)
    logger.info("TEST ZAVRŠEN")
    logger.info("Log fajl: %s", LOG_FILE)
    logger.info("JSON rezultati: %s", JSON_FILE)
    logger.info("=" * 90)


if __name__ == "__main__":
    main()