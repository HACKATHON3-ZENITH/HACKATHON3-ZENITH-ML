"""
model/segmenter.py — Segmentation explicite des utilisateurs.

TWIST 05 : Le système identifie deux profils distincts.

  • entrepreneur_actif  (~20 %) : a franchi le pas de l'action.
  • explorateur         (~80 %) : cherche encore sa voie.

Les recommandations DIFFÈRENT entre les deux profils.
La segmentation est EXPLICITE (fonction dédiée), pas implicite
ou émergente par hasard.
"""
from typing import Dict, Literal

import pandas as pd


UserSegment = Literal["entrepreneur_actif", "explorateur"]


# ---------------------------------------------------------------------------
# TWIST 05 — Fonction de segmentation individuelle
# ---------------------------------------------------------------------------

def segment_user(user_interactions: pd.DataFrame) -> UserSegment:
    """
    TWIST 05 : Classe un utilisateur dans l'un des deux segments.

    Règles de classification :
      1. Si l'utilisateur a lancé au moins une entreprise
         (business_launched == 1) → entrepreneur_actif.
      2. Sinon, si son taux d'actions terrain complétées ≥ 30 %
         → entrepreneur_actif.
      3. Sinon → explorateur.

    Args:
        user_interactions: DataFrame des interactions d'UN seul utilisateur
                           (colonnes attendues : action_completed,
                           business_launched).

    Returns:
        "entrepreneur_actif" ou "explorateur"
    """
    if user_interactions.empty:
        return "explorateur"

    # Critère 1 : a lancé un business
    has_launched = (user_interactions["business_launched"] == 1).any()
    if has_launched:
        return "entrepreneur_actif"

    # Critère 2 : taux d'actions ≥ 30 %
    action_rate = user_interactions["action_completed"].mean()
    if action_rate >= 0.3:
        return "entrepreneur_actif"

    return "explorateur"


# ---------------------------------------------------------------------------
# TWIST 05 — Segmentation de l'ensemble des utilisateurs
# ---------------------------------------------------------------------------

def segment_all_users(interactions_df: pd.DataFrame) -> Dict[str, UserSegment]:
    """
    Segmente TOUS les utilisateurs du jeu de données.

    Returns:
        Dictionnaire {learner_id: segment}.
    """
    segments: Dict[str, UserSegment] = {}
    for learner_id, group in interactions_df.groupby("learner_id"):
        segments[str(learner_id)] = segment_user(group)
    return segments


def get_segment_stats(segments: Dict[str, UserSegment]) -> Dict[str, int]:
    """Statistiques de répartition des segments."""
    counts = {"entrepreneur_actif": 0, "explorateur": 0}
    for seg in segments.values():
        counts[seg] += 1
    return counts


def validate_segment_distribution(segments: Dict[str, UserSegment]) -> bool:
    """
    TWIST 05 — Vérifie que la segmentation produit bien deux groupes
    distincts avec une répartition proche de ~20% actifs / ~80% explorateurs.

    Lève un warning si le ratio dévie fortement, mais ne bloque pas
    l'exécution (les données réelles peuvent varier).
    """
    import logging
    logger = logging.getLogger(__name__)
    stats = get_segment_stats(segments)
    total = sum(stats.values())
    if total == 0:
        return False
    ratio_actif = stats["entrepreneur_actif"] / total
    logger.info(
        "T05 — Répartition : %d actifs (%.0f%%) / %d explorateurs (%.0f%%)",
        stats["entrepreneur_actif"], ratio_actif * 100,
        stats["explorateur"], (1 - ratio_actif) * 100,
    )
    if ratio_actif < 0.05 or ratio_actif > 0.60:
        logger.warning("T05 — Répartition anormale ! Vérifier les critères.")
    return True
