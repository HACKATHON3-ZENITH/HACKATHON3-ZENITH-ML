"""
main.py — API FastAPI du système de recommandation Zenith Learn.

Endpoints exposés :
  GET  /health
  GET  /api/v1/recommendations/{learner_id}
  GET  /api/v1/recommendations/{learner_id}/explain/{course_id}
  GET  /api/v1/courses/{course_id}/insights
  GET  /api/v1/segments/{learner_id}
  GET  /api/v1/segments/stats

Twists traçables :
  T01 : /recommendations → score = alpha * engagement + beta * complétion
  T02 : Aucun endpoint de notification/relance/badge/streak n'existe.
  T03 : La complétion est ajustée par la durée (preprocessor).
  T04 : action_completed et business_launched exposés séparément.
  T05 : /segments → segmentation explicite.
  T06 : /insights et /explain → explications basées sur les vraies variables.
  T07 : /rankings/top-learners → classement du potentiel de réussite.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.loader import load_course_metadata, load_learner_ratings
from data.preprocessor import preprocess_interactions
from model.recommender import ZenithRecommender
from model.explainer import RecommendationExplainer
from model.segmenter import get_segment_stats

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zenith-ml")

# ──────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────
recommender: Optional[ZenithRecommender] = None
explainer: Optional[RecommendationExplainer] = None


# ──────────────────────────────────────────────────────────────────────
# Startup : chargement des données et entraînement du modèle
# ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, explainer

    logger.info("═" * 60)
    logger.info("  ZENITH LEARN — Recommendation Engine")
    logger.info("═" * 60)

    t0 = time.time()

    # 1. Charger les données brutes
    logger.info("Chargement des données…")
    courses_df = load_course_metadata()
    ratings_df = load_learner_ratings()
    logger.info(
        "  → %d cours, %d interactions, %d apprenants",
        len(courses_df),
        len(ratings_df),
        ratings_df["learner_id"].nunique(),
    )

    # 2. Préprocesser (T01, T03, T04)
    logger.info("Préprocessing (T01 engagement, T03 durée, T04 actions)…")
    interactions_df = preprocess_interactions(ratings_df, courses_df)

    # 3. Entraîner le modèle
    logger.info("Entraînement du recommandeur hybride…")
    recommender = ZenithRecommender(
        alpha_engagement=0.55,
        beta_completion=0.45,
        k_neighbors=20,
    )
    recommender.fit(interactions_df, courses_df)

    # 4. Initialiser l'explainer (T06)
    explainer = RecommendationExplainer(recommender)

    # 5. Stats
    seg_stats = get_segment_stats(recommender.user_segments)
    elapsed = time.time() - t0
    logger.info("  → Segments : %s", seg_stats)
    logger.info("  → Modèle prêt en %.2f secondes.", elapsed)
    logger.info("═" * 60)

    yield  # Application is running

    logger.info("Arrêt du moteur de recommandation.")


# ──────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Zenith Learn — Recommendation API",
    description=(
        "Système de recommandation éthique pour la plateforme Zenith Learn. "
        "Aucun mécanisme de rétention artificielle (T02)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Pydantic response models
# ──────────────────────────────────────────────────────────────────────
class CourseRecommendation(BaseModel):
    course_id: str
    titre: str
    categorie: str
    niveau: str
    duree_heures: float
    final_score: float
    detail: Dict


class RecommendationResponse(BaseModel):
    learner_id: str
    segment: str
    top_n: int
    recommendations: List[CourseRecommendation]
    twist_info: Dict


class SegmentResponse(BaseModel):
    learner_id: str
    segment: str


class SegmentStatsResponse(BaseModel):
    entrepreneur_actif: int
    explorateur: int
    total: int


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Vérification de santé de l'API."""
    return {
        "status": "ok",
        "model_loaded": recommender is not None and recommender._fitted,
    }


# ── T01 + T03 + T05 : Recommandations ────────────────────────────────

@app.get(
    "/api/v1/recommendations/{learner_id}",
    response_model=RecommendationResponse,
)
def get_recommendations(
    learner_id: str,
    top_n: int = Query(default=10, ge=1, le=25),
):
    """
    TWIST 01 : Score = alpha * engagement + beta * complétion.
    TWIST 03 : Complétion ajustée par la durée.
    TWIST 05 : Recommandations différenciées par segment.

    Retourne les top-N recommandations pour un apprenant.
    """
    if recommender is None or not recommender._fitted:
        raise HTTPException(503, "Modèle non chargé.")

    recs = recommender.recommend(learner_id, top_n=top_n)
    segment = recommender.get_user_segment(learner_id)

    return RecommendationResponse(
        learner_id=learner_id,
        segment=segment,
        top_n=top_n,
        recommendations=[CourseRecommendation(**r) for r in recs],
        twist_info={
            "T01": (
                f"Score = {recommender.ALPHA} × engagement "
                f"+ {recommender.BETA} × complétion"
            ),
            "T03": "Complétion normalisée par log(1+durée)/log(1+médiane)",
            "T05": f"Segment '{segment}' appliqué",
        },
    )


# ── T06 : Explication pour un apprenant ───────────────────────────────

@app.get("/api/v1/recommendations/{learner_id}/explain/{course_id}")
def explain_recommendation(learner_id: str, course_id: str):
    """
    TWIST 06 : Explication détaillée de pourquoi un cours est (ou non)
    recommandé à un apprenant spécifique.

    Basée sur les VRAIES variables du modèle.
    """
    if explainer is None:
        raise HTTPException(503, "Modèle non chargé.")

    try:
        return explainer.explain_recommendation_for_user(learner_id, course_id)
    except Exception as e:
        raise HTTPException(400, str(e))


# ── T06 : Insights pour les créateurs de cours ───────────────────────

@app.get("/api/v1/courses/{course_id}/insights")
def course_insights(course_id: str):
    """
    TWIST 06 : Explication lisible pour un créateur de contenu.

    Pourquoi ce cours est recommandé ou non, quels facteurs jouent
    en sa faveur/défaveur. Basé sur les vraies métriques du modèle.
    """
    if explainer is None:
        raise HTTPException(503, "Modèle non chargé.")

    result = explainer.explain_course_for_creator(course_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


# ── T05 : Segmentation ───────────────────────────────────────────────

@app.get(
    "/api/v1/segments/stats",
    response_model=SegmentStatsResponse,
)
def segment_stats():
    """
    TWIST 05 : Statistiques de répartition des segments.
    """
    if recommender is None:
        raise HTTPException(503, "Modèle non chargé.")
    stats = get_segment_stats(recommender.user_segments)
    return SegmentStatsResponse(
        entrepreneur_actif=stats["entrepreneur_actif"],
        explorateur=stats["explorateur"],
        total=stats["entrepreneur_actif"] + stats["explorateur"],
    )


@app.get(
    "/api/v1/segments/{learner_id}",
    response_model=SegmentResponse,
)
def get_segment(learner_id: str):
    """
    TWIST 05 : Retourne le segment explicite d'un apprenant.
    """
    if recommender is None:
        raise HTTPException(503, "Modèle non chargé.")
    segment = recommender.get_user_segment(learner_id)
    return SegmentResponse(learner_id=learner_id, segment=segment)


# ── T07 : Classement du Potentiel de Réussite ────────────────────────

class LearnerRankingEntry(BaseModel):
    learner_id: str
    score: float
    metrics: Dict
    explanation: Optional[Dict] = None

@app.get(
    "/api/v1/rankings/top-learners",
    response_model=List[LearnerRankingEntry],
)
def get_top_learners(
    top_n: int = Query(default=100, ge=1, le=1000),
    include_explanations: bool = Query(default=True),
):
    """
    TWIST 07 : Retourne les top-N apprenants les plus susceptibles de réussir.
    Inclut des explications basées sur les variables réelles (Engagement, Action, Segment).
    """
    if recommender is None or explainer is None:
        raise HTTPException(503, "Modèle non chargé.")

    rankings = recommender.rank_learners_by_success_potential(top_n=top_n)
    
    response = []
    for entry in rankings:
        rank_entry = LearnerRankingEntry(**entry)
        if include_explanations:
            rank_entry.explanation = explainer.explain_learner_success(entry["learner_id"])
        response.append(rank_entry)
        
    return response



# ──────────────────────────────────────────────────────────────────────
# TWIST 02 — AUCUN mécanisme de rétention artificielle
#
# Ce service de recommandation n'implémente AUCUN des mécanismes
# suivants, par décision de design explicite :
#   ✗ Notifications push
#   ✗ Emails de relance automatique
#   ✗ Badges ou récompenses virtuelles
#   ✗ Streaks ou séries quotidiennes
#   ✗ Compteurs de connexion consécutive
#   ✗ Tout autre dark pattern de rétention
#
# Le système recommande des cours pertinents. Il ne relance pas.
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
