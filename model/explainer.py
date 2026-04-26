"""
model/explainer.py — Explications lisibles pour les créateurs de cours.

TWIST 06 : Pour chaque cours, le système produit une explication basée
           sur les VRAIES variables du modèle (pas du texte générique).
           Un endpoint dédié expose ces explications.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .recommender import ZenithRecommender


class RecommendationExplainer:
    """
    TWIST 06 : Génère des explications lisibles par les créateurs de contenu.

    Chaque explication est construite à partir des vrais facteurs du modèle :
      - Score d'engagement moyen des apprenants
      - Score de complétion ajusté (durée neutralisée)
      - Taux d'action terrain
      - Taux de lancement d'entreprise
      - Impact du segment dominant
    """

    def __init__(self, recommender: ZenithRecommender):
        self.recommender = recommender

    # ------------------------------------------------------------------
    # TWIST 06 — Explication pour un créateur (vue globale d'un cours)
    # ------------------------------------------------------------------

    def explain_course_for_creator(self, course_id: str) -> Dict:
        """
        TWIST 06 : Produit une explication complète pour un créateur de cours.

        Retourne un dict structuré contenant :
          - Les métriques réelles du modèle pour ce cours
          - Les facteurs positifs et négatifs
          - Un texte d'explication lisible
        """
        rec = self.recommender
        if not rec._fitted:
            raise RuntimeError("Le modèle n'a pas été entraîné.")

        course_row = rec.courses_df[rec.courses_df["course_id"] == course_id]
        if course_row.empty:
            return {"error": f"Cours {course_id} introuvable."}

        meta = course_row.iloc[0]

        # Métriques réelles du modèle
        avg_engagement = rec.course_avg_engagement.get(course_id, 0.0)
        action_rate = rec.course_action_rates.get(course_id, 0.0)
        business_rate = rec.course_business_rates.get(course_id, 0.0)

        # Complétion ajustée moyenne pour ce cours (T03)
        course_interactions = rec.interactions_df[
            rec.interactions_df["course_id"] == course_id
        ]
        avg_adj_completion = float(
            course_interactions["adjusted_completion"].mean()
        ) if not course_interactions.empty else 0.0

        avg_raw_completion = float(
            course_interactions["completion_rate"].mean()
        ) if not course_interactions.empty else 0.0

        # Score que ce cours obtiendrait pour un utilisateur « médian »
        median_score = (
            rec.ALPHA * avg_engagement + rec.BETA * avg_adj_completion
        )

        # Analyse des forces et faiblesses
        strengths = []
        weaknesses = []

        if avg_engagement >= 0.6:
            strengths.append(
                f"Engagement élevé ({avg_engagement:.2f}) — "
                f"les apprenants apprécient ce cours et y investissent du temps."
            )
        elif avg_engagement < 0.4:
            weaknesses.append(
                f"Engagement faible ({avg_engagement:.2f}) — "
                f"les apprenants donnent des notes basses ou passent peu de temps."
            )

        if avg_adj_completion >= 0.7:
            strengths.append(
                f"Complétion ajustée élevée ({avg_adj_completion:.2f}) — "
                f"les apprenants terminent ce cours malgré sa durée "
                f"({meta['duree_heures']:.1f}h)."
            )
        elif avg_adj_completion < 0.4:
            weaknesses.append(
                f"Complétion ajustée faible ({avg_adj_completion:.2f}) — "
                f"beaucoup d'apprenants abandonnent avant la fin."
            )

        if action_rate >= 0.3:
            strengths.append(
                f"Bon taux d'action terrain ({action_rate:.0%}) — "
                f"ce cours mène à des actions concrètes."
            )
        elif action_rate < 0.1:
            weaknesses.append(
                f"Taux d'action terrain très faible ({action_rate:.0%}) — "
                f"peu d'apprenants passent à l'action après ce cours."
            )

        if business_rate >= 0.1:
            strengths.append(
                f"Taux de création d'entreprise notable ({business_rate:.0%}) — "
                f"ce cours contribue au passage à l'entrepreneuriat."
            )

        # Durée vs complétion brute (signal T03)
        if meta["duree_heures"] >= 5 and avg_raw_completion >= 0.6:
            strengths.append(
                f"Cours long ({meta['duree_heures']:.1f}h) bien complété "
                f"({avg_raw_completion:.0%}) — signe de contenu de qualité."
            )
        elif meta["duree_heures"] <= 2 and avg_raw_completion >= 0.9:
            weaknesses.append(
                f"Cours très court ({meta['duree_heures']:.1f}h) complété "
                f"à {avg_raw_completion:.0%} — la complétion élevée peut être "
                f"due à la brièveté, pas nécessairement à la qualité. "
                f"(neutralisé par l'ajustement T03)"
            )

        # Construction du texte lisible
        explanation_parts = []
        explanation_parts.append(
            f"📊 Score de recommandation médian : {median_score:.2f}/1.00"
        )
        explanation_parts.append(
            f"   → Contribution engagement : "
            f"{rec.ALPHA:.0%} × {avg_engagement:.2f} = "
            f"{rec.ALPHA * avg_engagement:.3f}"
        )
        explanation_parts.append(
            f"   → Contribution complétion : "
            f"{rec.BETA:.0%} × {avg_adj_completion:.2f} = "
            f"{rec.BETA * avg_adj_completion:.3f}"
        )

        if strengths:
            explanation_parts.append("\n✅ Forces :")
            for s in strengths:
                explanation_parts.append(f"  • {s}")

        if weaknesses:
            explanation_parts.append("\n⚠️ Axes d'amélioration :")
            for w in weaknesses:
                explanation_parts.append(f"  • {w}")

        # Qui voit ce cours ?
        # Simuler combien d'utilisateurs recevraient cette recommandation
        nb_would_recommend = self._count_potential_recommendations(course_id)
        total_users = len(rec.user_ids)
        explanation_parts.append(
            f"\n👥 Visibilité estimée : ce cours apparaîtrait dans le top 10 "
            f"de ~{nb_would_recommend}/{total_users} utilisateurs "
            f"({nb_would_recommend/max(total_users,1):.0%})."
        )

        return {
            "course_id": course_id,
            "titre": meta["titre"],
            "categorie": meta["categorie"],
            "niveau": meta["niveau"],
            "duree_heures": float(meta["duree_heures"]),
            "metrics": {
                "avg_engagement_score": round(avg_engagement, 4),
                "avg_adjusted_completion": round(avg_adj_completion, 4),
                "avg_raw_completion": round(avg_raw_completion, 4),
                "action_rate": round(action_rate, 4),
                "business_rate": round(business_rate, 4),
                "median_recommendation_score": round(median_score, 4),
                "alpha_engagement": rec.ALPHA,
                "beta_completion": rec.BETA,
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "explanation_text": "\n".join(explanation_parts),
            "visibility_estimate": {
                "would_appear_in_top10_for": nb_would_recommend,
                "total_users": total_users,
            },
        }

    def _count_potential_recommendations(
        self, course_id: str, top_n: int = 10
    ) -> int:
        """
        Estime combien d'utilisateurs recevraient ce cours dans leur top N.
        (échantillon pour performance)
        """
        rec = self.recommender
        count = 0
        # Échantillonner pour ne pas être trop lent
        sample_users = rec.user_ids[:200]
        for uid in sample_users:
            recs = rec.recommend(uid, top_n=top_n)
            if any(r["course_id"] == course_id for r in recs):
                count += 1
        # Extrapoler
        ratio = count / max(len(sample_users), 1)
        return int(ratio * len(rec.user_ids))

    # ------------------------------------------------------------------
    # TWIST 06 — Explication pour un apprenant (pourquoi CE cours)
    # ------------------------------------------------------------------

    def explain_recommendation_for_user(
        self, learner_id: str, course_id: str
    ) -> Dict:
        """
        TWIST 06 : Explique pourquoi un cours est recommandé (ou non)
        à un utilisateur spécifique.

        Retourne les facteurs réels du modèle et un texte lisible.
        """
        rec = self.recommender
        if not rec._fitted:
            raise RuntimeError("Le modèle n'a pas été entraîné.")

        segment = rec.get_user_segment(learner_id)

        eng = rec._predict_engagement(learner_id, course_id)
        comp = rec._predict_completion(learner_id, course_id)
        content_sim = rec._content_similarity(learner_id, course_id)

        engagement_pred = 0.7 * eng + 0.3 * content_sim
        completion_pred = comp

        final_score, detail = rec.compute_final_score(
            engagement_pred, completion_pred, segment, course_id
        )

        course_row = rec.courses_df[rec.courses_df["course_id"] == course_id]
        titre = course_row.iloc[0]["titre"] if not course_row.empty else course_id

        # Texte d'explication
        parts = [
            f"Cours : « {titre} »",
            f"Score final : {final_score:.3f}",
            f"",
            f"Détail du score :",
            f"  • Engagement prédit : {engagement_pred:.3f} "
            f"(poids {rec.ALPHA:.0%} → {rec.ALPHA * engagement_pred:.3f})",
            f"  • Complétion prédite (ajustée durée) : {completion_pred:.3f} "
            f"(poids {rec.BETA:.0%} → {rec.BETA * completion_pred:.3f})",
            f"  • Similarité contenu : {content_sim:.3f}",
            f"  • Segment : {segment} → {detail['segment_reason']}",
            f"  • Taux action terrain du cours : "
            f"{detail['course_action_rate']:.0%}",
            f"  • Taux création entreprise du cours : "
            f"{detail['course_business_rate']:.0%}",
        ]

        return {
            "learner_id": learner_id,
            "course_id": course_id,
            "titre": titre,
            "segment": segment,
            "final_score": round(final_score, 4),
            "detail": detail,
            "content_similarity": round(content_sim, 4),
            "explanation_text": "\n".join(parts),
        }
