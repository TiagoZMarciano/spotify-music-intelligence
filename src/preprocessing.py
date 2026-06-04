from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"

IDENTIFIER_COLUMNS = ["track_id", "artists", "album_name", "track_name"]
TARGET_COLUMN = "track_genre"

AUDIO_FEATURES = [
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

MODEL_FEATURES = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_tracks(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the Spotify tracks dataset from disk.

    This function keeps file loading in one place, so notebooks and scripts can
    reuse the same dataset path instead of hardcoding it multiple times.
    """
    return pd.read_csv(path)


def clean_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat the raw dataset with the first project-level cleaning rules.

    The goal is not to over-clean the data at this stage. We only remove fields
    or rows that would break the core analysis, genre modeling, or recommender.
    """
    cleaned = df.copy()

    # Drop the exported index column from the Kaggle CSV.
    # It does not represent a real Spotify/audio attribute.
    if "Unnamed: 0" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Unnamed: 0"])

    # Convert the boolean explicit flag into 0/1 so it can be used by models.
    cleaned["explicit"] = cleaned["explicit"].astype(int)

    # Drop rows without the minimum identifiers needed for this project:
    # target genre for modeling, track id for deduplication, and track name for
    # readable analysis/recommendations.
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN, "track_id", "track_name"])

    # The same track can appear more than once inside the same genre.
    # Keeping only one record per track/genre pair avoids overweighting duplicates.
    cleaned = cleaned.drop_duplicates(subset=["track_id", TARGET_COLUMN])

    # Reset the index after row removals to keep downstream joins and selections clean.
    return cleaned.reset_index(drop=True)


def select_feature_matrix(
    df: pd.DataFrame,
    features: Iterable[str] = MODEL_FEATURES,
) -> pd.DataFrame:
    """
    Select the numeric feature matrix used by machine learning models.

    A stable feature order is important because scikit-learn models learn from
    arrays where column position matters.
    """
    return df.loc[:, list(features)].copy()


def scale_features(
    df: pd.DataFrame,
    features: Iterable[str] = MODEL_FEATURES,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numeric audio features for PCA, clustering, and distance models.

    Scaling is necessary because features use different ranges. For example,
    tempo is measured in BPM, duration is measured in milliseconds, and features
    such as energy or valence are already between 0 and 1.
    """
    feature_names = list(features)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_names])
    scaled = pd.DataFrame(scaled_values, columns=feature_names, index=df.index)
    return scaled, scaler


def make_train_test_split(
    df: pd.DataFrame,
    features: Iterable[str] = MODEL_FEATURES,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Create a stratified train/test split for genre classification.

    Stratification preserves the genre distribution in both train and test sets,
    which is important because this is a multi-class classification problem.
    """
    x = select_feature_matrix(df, features)
    y = df[TARGET_COLUMN]

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
