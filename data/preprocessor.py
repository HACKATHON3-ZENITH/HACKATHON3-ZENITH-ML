"""
data/preprocessor.py — Feature engineering pour le système de recommandation.

Ce module est directement traçable pour :
  TWIST 01 : Calcul séparé des scores d'engagement et de complétion.
  TWIST 03 : Normalisation de la complétion par la durée du cours.
  TWIST 04 : Les signaux action_completee et entreprise_lancee restent
             des colonnes séparées, jamais fusionnées entre elles ni avec
             la complétion.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TWIST 03 — Neutraliser le biais de durée
# ---------------------------------------------------------------------------

def adjust_completion_for_duration(
    completion_rate: float,
    duration_hours: float,
    median_duration: float = 4.3,
) -> float:
    """
    TWIST 03 : Ajuste le taux de complétion pour que la durée du cours
    n'avantage ni ne pénalise la recommandation.

    Un cours de 1 h complété à 100 % ne doit pas scorer plus haut qu'un
    cours de 7 h complété à 80 %. On pondère par log(1 + durée) / log(1 + médiane)
    pour donner un léger crédit aux cours longs sans écraser les courts.

    Args:
        completion_rate: Taux brut (0-1).
        duration_hours:  Durée du cours en heures.
        median_duration: Durée médiane du catalogue (point de référence neutre).

    Returns:
        Taux de complétion ajusté.
    """
    if duration_hours <= 0:
        return completion_rate
    adjustment_factor = np.log1p(duration_hours) / np.log1p(median_duration)
    return float(completion_rate * adjustment_factor)


# ---------------------------------------------------------------------------
# TWIST 01 — Score d'engagement (composante distincte)
# ---------------------------------------------------------------------------

def compute_engagement_score(
    rating: float,
    time_spent_hours: float,
    expected_duration_hours: float,
) -> float:
    """
    TWIST 01 (composante engagement) : Calcule un score d'engagement à
    partir du rating ET du temps investi.

    L'engagement est un signal SÉPARÉ de la complétion. Il capture
    « l'utilisateur a-t-il apprécié et investi du temps ? ».

    Args:
        rating: Note donnée par l'utilisateur (1-5).
        time_spent_hours: Temps réellement passé.
        expected_duration_hours: Durée prévue du cours.

    Returns:
        Score d'engagement normalisé entre 0 et 1.
    """
    # Normaliser le rating sur [0, 1]
    norm_rating = (rating - 1.0) / 4.0

    # Ratio temps investi / temps attendu, plafonné à 1.5
    time_ratio = min(time_spent_hours / max(expected_duration_hours, 0.1), 1.5)
    norm_time = time_ratio / 1.5  # ramener sur [0, 1]

    # Combinaison pondérée : le rating pèse 60 %, le temps 40 %
    return float(0.6 * norm_rating + 0.4 * norm_time)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def preprocess_interactions(
    ratings_df: pd.DataFrame,
    courses_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusionne les interactions avec les métadonnées et ajoute les features.

    Colonnes ajoutées :
      - engagement_score      (TWIST 01)
      - adjusted_completion   (TWIST 03)
      - action_completed      (TWIST 04 — signal séparé, binaire)
      - business_launched     (TWIST 04 — signal séparé, binaire)

    Les deux signaux d'action (T04) ne sont JAMAIS fusionnés entre eux
    ni avec adjusted_completion.
    """
    merged = ratings_df.merge(
        courses_df[["course_id", "duree_heures", "categorie", "niveau",
                     "stade_entrepreneur"]],
        on="course_id",
        how="left",
    )

    # Calcul de la médiane de durée sur le catalogue réel
    median_dur = courses_df["duree_heures"].median()

    # TWIST 04 : signaux d'action convertis en binaire (colonnes SÉPARÉES)
    merged["action_completed"] = (
        merged["action_completee"].str.strip().str.lower() == "oui"
    ).astype(int)
    merged["business_launched"] = (
        merged["entreprise_lancee"].str.strip().str.lower() == "oui"
    ).astype(int)

    # TWIST 03 : complétion ajustée par la durée
    merged["adjusted_completion"] = merged.apply(
        lambda r: adjust_completion_for_duration(
            r["completion_rate"], r["duree_heures"], median_dur
        ),
        axis=1,
    )

    # TWIST 01 : score d'engagement
    merged["engagement_score"] = merged.apply(
        lambda r: compute_engagement_score(
            r["rating"], r["time_spent_hours"], r["duree_heures"]
        ),
        axis=1,
    )

    return merged
