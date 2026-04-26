"""
data/loader.py — Chargement des données brutes Zenith Learn.

Charge les métadonnées des cours et les interactions apprenants-cours
depuis les fichiers du dossier docsML.
"""
import json
import os
import pandas as pd
from pathlib import Path


def _get_data_dir() -> Path:
    """Resolve the data directory path."""
    env_path = os.environ.get("DATA_DIR")
    if env_path:
        return Path(env_path)
    # Default: ../docsML relative to zenith-ml/
    return Path(__file__).parent.parent.parent / "docsML"


def load_course_metadata() -> pd.DataFrame:
    """
    Load course metadata from cours_metadata.json.

    Returns a DataFrame with columns:
        course_id, titre, categorie, duree_heures, niveau,
        stade_entrepreneur, taux_completion_moyen, note_moyenne,
        nb_inscrits, actions_terrain_associees, prerequis_course_id
    """
    data_dir = _get_data_dir()
    filepath = data_dir / "cours_metadata.json"
    with open(filepath, "r", encoding="utf-8") as f:
        courses = json.load(f)
    return pd.DataFrame(courses)


def load_learner_ratings() -> pd.DataFrame:
    """
    Load learner-course interaction data from learner_ratings.csv.

    Returns a DataFrame with columns:
        learner_id, course_id, completion_rate, time_spent_hours,
        rating, action_completee, entreprise_lancee
    """
    data_dir = _get_data_dir()
    filepath = data_dir / "learner_ratings.csv"
    df = pd.read_csv(filepath)
    # Clean up potential \r characters
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df
