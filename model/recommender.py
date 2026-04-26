"""
model/recommender.py — Moteur de recommandation hybride Zenith Learn.

Twists adressés directement dans ce module :
  TWIST 01 : Le score final = alpha * engagement + beta * complétion.
             Les deux signaux sont des entrées DISTINCTES, calculées
             séparément, et combinées dans compute_final_score().
  TWIST 03 : La complétion utilisée est TOUJOURS adjusted_completion
             (normalisée par la durée dans preprocessor.py).
  TWIST 05 : Les recommandations sont ajustées par segment utilisateur
             dans compute_final_score().
  TWIST 08 : Neutralisation du biais des apprenants les plus disponibles.
             Le modèle identifie les outliers à haut débit et normalise
             leurs métriques pour éviter qu'ils ne dominent les scores.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .segmenter import UserSegment, segment_all_users, segment_user

logger = logging.getLogger(__name__)


class ZenithRecommender:
    """
    Recommandeur hybride : filtrage collaboratif + contenu.

    Architecture :
      1. Matrice utilisateur-cours (engagement) → prédiction d'engagement
         via KNN collaboratif.
      2. Matrice utilisateur-cours (complétion ajustée) → prédiction de
         complétion via KNN collaboratif.
      3. Vecteurs de contenu des cours → similarité contenu-préférence.
      4. Score final = alpha * engagement_pred + beta * completion_pred
         + ajustement de segment.
    """

    def __init__(
        self,
        alpha_engagement: float = 0.55,
        beta_completion: float = 0.45,
        k_neighbors: int = 20,
    ):
        # ── TWIST 01 : poids EXPLICITES pour les deux signaux ──
        assert alpha_engagement > 0, "T01 : le poids engagement doit être > 0"
        assert beta_completion > 0, "T01 : le poids complétion doit être > 0"
        self.ALPHA = alpha_engagement
        self.BETA = beta_completion
        self.K = k_neighbors

        # Données internes
        self.interactions_df: Optional[pd.DataFrame] = None
        self.courses_df: Optional[pd.DataFrame] = None
        self.user_ids: List[str] = []
        self.course_ids: List[str] = []

        # Matrices pour le filtrage collaboratif
        self.engagement_matrix: Optional[np.ndarray] = None
        self.completion_matrix: Optional[np.ndarray] = None
        self.user_similarity: Optional[np.ndarray] = None

        # Contenu
        self.course_feature_vectors: Optional[np.ndarray] = None
        self.user_profiles: Dict[str, np.ndarray] = {}

        # Segments (T05)
        self.user_segments: Dict[str, UserSegment] = {}

        # Statistiques par cours (T04, T06)
        self.course_action_rates: Dict[str, float] = {}
        self.course_business_rates: Dict[str, float] = {}
        self.course_avg_engagement: Dict[str, float] = {}

        # TWIST 08 : Neutralisation du biais (Outliers)
        self.user_bias_factors: Dict[str, float] = {}
        self.outlier_user_ids: List[str] = []

        self._fitted = False

    # ==================================================================
    # ENTRAÎNEMENT
    # ==================================================================

    def fit(self, interactions_df: pd.DataFrame, courses_df: pd.DataFrame):
        """
        Entraîne le recommandeur sur les données préprocessées.

        interactions_df doit contenir les colonnes :
            learner_id, course_id, engagement_score, adjusted_completion,
            action_completed, business_launched, rating, categorie, niveau
        courses_df : métadonnées originales des cours.
        """
        self.interactions_df = interactions_df.copy()
        self.courses_df = courses_df.copy()

        self.user_ids = sorted(interactions_df["learner_id"].unique().tolist())
        self.course_ids = sorted(courses_df["course_id"].unique().tolist())

        self._build_collaborative_matrices()
        self._compute_user_similarity()
        self._build_course_features()
        self._build_user_preference_profiles()
        self._compute_course_stats()

        # TWIST 08 : Détection des outliers avant segmentation
        self._detect_outliers()

        # TWIST 05 : segmentation explicite (mise à jour pour T08)
        self.user_segments = segment_all_users(interactions_df)
        logger.info(
            "Segments — actifs : %d, explorateurs : %d",
            sum(1 for s in self.user_segments.values() if s == "entrepreneur_actif"),
            sum(1 for s in self.user_segments.values() if s == "explorateur"),
        )

        self._fitted = True
        logger.info("Modèle entraîné sur %d utilisateurs × %d cours.",
                     len(self.user_ids), len(self.course_ids))

    # ------------------------------------------------------------------
    # Construction des matrices collaboratives
    # ------------------------------------------------------------------

    def _build_collaborative_matrices(self):
        """
        Construit deux matrices utilisateur × cours :
          - engagement_matrix  (TWIST 01 signal engagement)
          - completion_matrix  (TWIST 01 signal complétion, TWIST 03 ajusté)
        """
        n_users = len(self.user_ids)
        n_courses = len(self.course_ids)
        user_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        course_idx = {cid: i for i, cid in enumerate(self.course_ids)}

        self.engagement_matrix = np.full((n_users, n_courses), np.nan)
        self.completion_matrix = np.full((n_users, n_courses), np.nan)

        for _, row in self.interactions_df.iterrows():
            ui = user_idx.get(row["learner_id"])
            ci = course_idx.get(row["course_id"])
            if ui is not None and ci is not None:
                self.engagement_matrix[ui, ci] = row["engagement_score"]
                # TWIST 03 : on utilise adjusted_completion, pas completion_rate
                self.completion_matrix[ui, ci] = row["adjusted_completion"]

        # Remplacer les NaN par la moyenne de la colonne (average du cours)
        for mat in (self.engagement_matrix, self.completion_matrix):
            col_means = np.nanmean(mat, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            for j in range(n_courses):
                nan_mask = np.isnan(mat[:, j])
                mat[nan_mask, j] = col_means[j]

    def _compute_user_similarity(self):
        """Similarité cosinus entre utilisateurs (basée sur l'engagement)."""
        self.user_similarity = cosine_similarity(self.engagement_matrix)

    # ------------------------------------------------------------------
    # Features de contenu des cours
    # ------------------------------------------------------------------

    def _build_course_features(self):
        """
        Vectorise chaque cours à partir de ses métadonnées :
          catégorie (one-hot), niveau (ordinal), stade (one-hot),
          popularité (log), durée (log).
        """
        df = self.courses_df.copy()

        # Catégorie : one-hot
        cat_dummies = pd.get_dummies(df["categorie"], prefix="cat")

        # Niveau : ordinal
        level_map = {"debutant": 0.0, "intermediaire": 0.5, "avance": 1.0}
        df["niveau_ord"] = df["niveau"].map(level_map).fillna(0.5)

        # Stade : one-hot
        stade_dummies = pd.get_dummies(df["stade_entrepreneur"], prefix="stade")

        # Popularité et durée normalisées
        df["log_inscrits"] = np.log1p(df["nb_inscrits"])
        df["log_duree"] = np.log1p(df["duree_heures"])

        scaler = MinMaxScaler()
        num_features = scaler.fit_transform(
            df[["niveau_ord", "log_inscrits", "log_duree"]]
        )

        feature_matrix = np.hstack([
            cat_dummies.values,
            stade_dummies.values,
            num_features,
        ])

        self.course_feature_vectors = feature_matrix
        self._course_feature_names = (
            list(cat_dummies.columns)
            + list(stade_dummies.columns)
            + ["niveau_ord", "log_inscrits", "log_duree"]
        )

    def _build_user_preference_profiles(self):
        """
        Pour chaque utilisateur, construit un vecteur de préférences
        = moyenne pondérée (par engagement_score) des vecteurs de cours
        avec lesquels il a interagi.
        """
        course_idx = {cid: i for i, cid in enumerate(self.course_ids)}

        for uid in self.user_ids:
            user_data = self.interactions_df[
                self.interactions_df["learner_id"] == uid
            ]
            if user_data.empty:
                self.user_profiles[uid] = np.zeros(
                    self.course_feature_vectors.shape[1]
                )
                continue

            weights = []
            vectors = []
            for _, row in user_data.iterrows():
                ci = course_idx.get(row["course_id"])
                if ci is not None:
                    weights.append(row["engagement_score"])
                    vectors.append(self.course_feature_vectors[ci])

            if not weights:
                self.user_profiles[uid] = np.zeros(
                    self.course_feature_vectors.shape[1]
                )
            else:
                w = np.array(weights).reshape(-1, 1)
                v = np.array(vectors)
                self.user_profiles[uid] = (v * w).sum(axis=0) / (w.sum() + 1e-9)

    # ------------------------------------------------------------------
    # Statistiques par cours (TWIST 04 signaux séparés, TWIST 06)
    # ------------------------------------------------------------------

    def _compute_course_stats(self):
        """
        TWIST 04 : Calcule les taux d'action et de création d'entreprise
        pour chaque cours SÉPARÉMENT (jamais fusionnés).
        """
        for cid, grp in self.interactions_df.groupby("course_id"):
            self.course_action_rates[cid] = float(grp["action_completed"].mean())
            self.course_business_rates[cid] = float(grp["business_launched"].mean())
            self.course_avg_engagement[cid] = float(grp["engagement_score"].mean())

    # ------------------------------------------------------------------
    # TWIST 08 — Neutralisation du biais (Outliers haut débit)
    # ------------------------------------------------------------------

    def _detect_outliers(self):
        """
        TWIST 08 : Identifie les apprenants dont le débit ou le temps de
        connexion est anormalement élevé.
        """
        # Calculer le volume total de complétion et d'engagement par utilisateur
        user_stats = self.interactions_df.groupby("learner_id").agg({
            "adjusted_completion": "sum",
            "engagement_score": "sum",
            "course_id": "count"
        }).rename(columns={"course_id": "n_courses"})

        # Métrique de "débit" : volume par cours
        user_stats["velocity"] = user_stats["adjusted_completion"] / user_stats["n_courses"]
        
        # Détection statistique (Mean + 2*STD)
        threshold_vol = user_stats["adjusted_completion"].mean() + 2 * user_stats["adjusted_completion"].std()
        threshold_vel = user_stats["velocity"].mean() + 2 * user_stats["velocity"].std()

        self.user_bias_factors = {}
        self.outlier_user_ids = []

        for uid, row in user_stats.iterrows():
            factor = 1.0
            is_outlier = False

            # Si volume trop élevé (apprenant qui a trop de temps libre)
            if row["adjusted_completion"] > threshold_vol:
                # Pénalité douce : amortir le surplus
                factor = min(factor, threshold_vol / row["adjusted_completion"])
                is_outlier = True
            
            # Si vitesse trop élevée (apprenant "trop rapide")
            if row["velocity"] > threshold_vel:
                factor = min(factor, threshold_vel / row["velocity"])
                is_outlier = True
            
            self.user_bias_factors[str(uid)] = factor
            if is_outlier:
                self.outlier_user_ids.append(str(uid))

        logger.info(
            "T08 — Outliers détectés : %d/%d (facteur moyen : %.2f)",
            len(self.outlier_user_ids),
            len(self.user_ids),
            np.mean(list(self.user_bias_factors.values()))
        )

    # ==================================================================
    # PRÉDICTION
    # ==================================================================

    def _predict_engagement(self, learner_id: str, course_id: str) -> float:
        """
        TWIST 01 (signal engagement) : Prédit l'engagement d'un utilisateur
        pour un cours non encore suivi via filtrage collaboratif (KNN).
        """
        if learner_id not in self.user_ids:
            return self.course_avg_engagement.get(course_id, 0.5)

        ui = self.user_ids.index(learner_id)
        ci = self.course_ids.index(course_id)

        # Trouver les K voisins les plus similaires ayant noté ce cours
        similarities = self.user_similarity[ui].copy()
        similarities[ui] = -1  # exclure soi-même

        # Pondération par similarité
        top_k_indices = np.argsort(similarities)[-self.K:]
        top_k_sims = similarities[top_k_indices]
        top_k_vals = self.engagement_matrix[top_k_indices, ci]

        denom = np.abs(top_k_sims).sum()
        if denom == 0:
            return self.course_avg_engagement.get(course_id, 0.5)

        return float(np.dot(top_k_sims, top_k_vals) / denom)

    def _predict_completion(self, learner_id: str, course_id: str) -> float:
        """
        TWIST 01 (signal complétion) + TWIST 03 (ajusté par durée) :
        Prédit le taux de complétion ajusté via filtrage collaboratif.
        """
        if learner_id not in self.user_ids:
            # Cold start : moyenne du cours
            ci = self.course_ids.index(course_id) if course_id in self.course_ids else None
            if ci is not None:
                return float(np.nanmean(self.completion_matrix[:, ci]))
            return 0.5

        ui = self.user_ids.index(learner_id)
        ci = self.course_ids.index(course_id)

        similarities = self.user_similarity[ui].copy()
        similarities[ui] = -1

        top_k_indices = np.argsort(similarities)[-self.K:]
        top_k_sims = similarities[top_k_indices]
        top_k_vals = self.completion_matrix[top_k_indices, ci]

        denom = np.abs(top_k_sims).sum()
        if denom == 0:
            return float(np.nanmean(self.completion_matrix[:, ci]))

        return float(np.dot(top_k_sims, top_k_vals) / denom)

    def _content_similarity(self, learner_id: str, course_id: str) -> float:
        """Similarité contenu entre le profil utilisateur et un cours."""
        if learner_id not in self.user_profiles:
            return 0.0
        ci = self.course_ids.index(course_id)
        user_vec = self.user_profiles[learner_id]
        course_vec = self.course_feature_vectors[ci]
        norm_u = np.linalg.norm(user_vec)
        norm_c = np.linalg.norm(course_vec)
        if norm_u == 0 or norm_c == 0:
            return 0.0
        return float(np.dot(user_vec, course_vec) / (norm_u * norm_c))

    # ------------------------------------------------------------------
    # TWIST 01 + TWIST 05 : Score final
    # ------------------------------------------------------------------

    def compute_final_score(
        self,
        engagement_pred: float,
        completion_pred: float,
        segment: UserSegment,
        course_id: str,
    ) -> Tuple[float, Dict]:
        """
        TWIST 01 : Score final = ALPHA * engagement + BETA * complétion.
        TWIST 05 : Ajustement selon le segment utilisateur.

        Returns:
            (score_final, détail_des_composantes)
        """
        # ── TWIST 01 : combinaison EXPLICITE des deux signaux ──
        base_score = self.ALPHA * engagement_pred + self.BETA * completion_pred

        # ── TWIST 08 : neutralisation du biais d'outlier ──
        # Si le profil actuel subit une correction, on l'applique ici
        # Note : On ne connaît pas l'ID ici si c'est une prédiction pure,
        # mais on peut passer un bias_factor optionnel si besoin.
        # Dans recommand(), on utilise l'id.

        # ── TWIST 05 : ajustement par segment ──
        segment_adjustment = 0.0
        segment_reason = ""

        if segment == "entrepreneur_actif":
            # Les entrepreneurs actifs valorisent les cours menant à l'action
            action_rate = self.course_action_rates.get(course_id, 0.0)
            business_rate = self.course_business_rates.get(course_id, 0.0)
            segment_adjustment = 0.08 * action_rate + 0.05 * business_rate
            segment_reason = (
                f"Entrepreneur actif → boost action terrain "
                f"(+{segment_adjustment:.3f})"
            )
        else:
            # Les explorateurs valorisent l'accessibilité et la découverte
            course_row = self.courses_df[
                self.courses_df["course_id"] == course_id
            ]
            if not course_row.empty:
                level = course_row.iloc[0]["niveau"]
                accessibility = 1.0 if level == "debutant" else (
                    0.6 if level == "intermediaire" else 0.3
                )
                segment_adjustment = 0.06 * accessibility
                segment_reason = (
                    f"Explorateur → boost accessibilité niveau={level} "
                    f"(+{segment_adjustment:.3f})"
                )

        final_score = base_score + segment_adjustment

        # ── TWIST 09 : Décomposition quantitative ligne par ligne ──
        contributions = [
            {"variable": "Engagement (Prédiction CF/Contenu)", "weight": self.ALPHA, "value": round(engagement_pred, 4), "contribution": round(self.ALPHA * engagement_pred, 4)},
            {"variable": "Complétion (Prédiction CF)", "weight": self.BETA, "value": round(completion_pred, 4), "contribution": round(self.BETA * completion_pred, 4)},
            {"variable": f"Ajustement Segment ({segment})", "weight": 1.0, "value": round(segment_adjustment, 4), "contribution": round(segment_adjustment, 4)}
        ]

        detail = {
            "engagement_pred": round(engagement_pred, 4),
            "engagement_weight": self.ALPHA,
            "engagement_contribution": round(self.ALPHA * engagement_pred, 4),
            "completion_pred": round(completion_pred, 4),
            "completion_weight": self.BETA,
            "completion_contribution": round(self.BETA * completion_pred, 4),
            "base_score": round(base_score, 4),
            "segment": segment,
            "segment_adjustment": round(segment_adjustment, 4),
            "segment_reason": segment_reason,
            "final_score": round(final_score, 4),
            "contributions": contributions,
            # TWIST 04 : signaux d'action exposés SÉPARÉMENT
            "course_action_rate": round(
                self.course_action_rates.get(course_id, 0.0), 4
            ),
            "course_business_rate": round(
                self.course_business_rates.get(course_id, 0.0), 4
            ),
        }

        return final_score, detail

    def compute_final_score_with_bias(
        self,
        engagement_pred: float,
        completion_pred: float,
        segment: UserSegment,
        course_id: str,
        learner_id: str,
    ) -> Tuple[float, Dict]:
        """Extension de compute_final_score incluant le TWIST 08."""
        score, detail = self.compute_final_score(engagement_pred, completion_pred, segment, course_id)
        
        # Application du TWIST 08
        bias_factor = self.user_bias_factors.get(learner_id, 1.0)
        if bias_factor < 1.0:
            original_score = score
            score *= bias_factor
            detail["final_score"] = round(score, 4)
            detail["twist_08_bias_correction"] = round(bias_factor, 4)
            detail["segment_reason"] += f" | T08: Correction disponibilité (x{bias_factor:.2f})"
            # TWIST 09 : Ajout de la pénalité de biais dans les contributions
            detail["contributions"].append({
                "variable": "Pénalité de disponibilité (Outlier T08)",
                "weight": "multiplicatif",
                "value": round(bias_factor, 4),
                "contribution": round(score - original_score, 4)
            })
        
        return score, detail

    # ==================================================================
    # RECOMMANDATION
    # ==================================================================

    def recommend(
        self,
        learner_id: str,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Génère les top-N recommandations pour un utilisateur.

        TWIST 01 : Le score combine engagement ET complétion.
        TWIST 03 : La complétion est ajustée par la durée.
        TWIST 05 : Les recommandations diffèrent selon le segment.

        Returns:
            Liste de dicts triés par score décroissant.
        """
        if not self._fitted:
            raise RuntimeError("Le modèle n'a pas été entraîné (appeler fit()).")

        # Cours déjà suivis par l'utilisateur
        taken = set()
        if self.interactions_df is not None:
            taken = set(
                self.interactions_df[
                    self.interactions_df["learner_id"] == learner_id
                ]["course_id"].tolist()
            )

        # Segment de l'utilisateur (T05)
        segment = self.user_segments.get(learner_id, "explorateur")

        candidates = []
        for cid in self.course_ids:
            if cid in taken:
                continue

            eng = self._predict_engagement(learner_id, cid)
            comp = self._predict_completion(learner_id, cid)
            content_sim = self._content_similarity(learner_id, cid)

            # Combiner CF et contenu pour les prédictions
            engagement_pred = 0.7 * eng + 0.3 * content_sim
            completion_pred = comp  # déjà ajusté par durée (T03)

            final_score, detail = self.compute_final_score_with_bias(
                engagement_pred, completion_pred, segment, cid, learner_id
            )

            # Enrichir avec les métadonnées du cours
            course_meta = self.courses_df[
                self.courses_df["course_id"] == cid
            ].iloc[0]

            candidates.append({
                "course_id": cid,
                "titre": course_meta["titre"],
                "categorie": course_meta["categorie"],
                "niveau": course_meta["niveau"],
                "duree_heures": float(course_meta["duree_heures"]),
                "final_score": round(final_score, 4),
                "detail": detail,
            })

        # Trier par score décroissant
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_n]

    def get_user_segment(self, learner_id: str) -> UserSegment:
        """Retourne le segment d'un utilisateur."""
        return self.user_segments.get(learner_id, "explorateur")

    # ==================================================================
    # TWIST 07 : Classement du Potentiel de Réussite
    # ==================================================================

    def rank_learners_by_success_potential(self, top_n: int = 100) -> List[Dict]:
        """
        TWIST 07 : Identifie les top-N apprenants les plus susceptibles de réussir.
        Le score de potentiel est distinct du score de recommandation de cours.
        """
        if not self._fitted:
            raise RuntimeError("Le modèle n'a pas été entraîné.")

        rankings = []
        for uid in self.user_ids:
            user_data = self.interactions_df[self.interactions_df["learner_id"] == uid]
            segment = self.get_user_segment(uid)
            
            # 1. Signal d'action (TWIST 04)
            action_count = user_data["action_completed"].sum()
            business_launched = user_data["business_launched"].any()
            
            # 2. Engagement moyen (TWIST 01)
            avg_engagement = user_data["engagement_score"].mean() if not user_data.empty else 0.0
            
            # 3. Complétion ajustée (TWIST 03)
            avg_completion = user_data["adjusted_completion"].mean() if not user_data.empty else 0.0
            
            # 4. Diversité des compétences (Nombre de catégories uniques explorées)
            categories_count = user_data["course_id"].nunique()
            
            # 5. TWIST 08 : Facteur de correction (Neutralisation disponibilité)
            bias_factor = self.user_bias_factors.get(uid, 1.0)
            
            # Calcul du Success Potential Score
            # Poids : Business (0.4) + Actions (0.3) + Engagement (0.15) + Segment (0.15)
            # Puis redressé par le biais T08
            # TWIST 09 : Décomposition pour le score de potentiel investisseur
            contributions = [
                {"variable": "Lancement d'entreprise", "weight": 0.40, "value": 1.0 if business_launched else 0.0, "contribution": 0.40 if business_launched else 0.0},
                {"variable": "Actions terrain accomplies", "weight": 0.30, "value": round(min(action_count / 5.0, 1.0), 4), "contribution": round(0.30 * min(action_count / 5.0, 1.0), 4)},
                {"variable": "Engagement moyen", "weight": 0.15, "value": round(avg_engagement, 4), "contribution": round(0.15 * avg_engagement, 4)},
                {"variable": f"Segment ({segment})", "weight": 0.15, "value": 1.0 if segment == "entrepreneur_actif" else 0.4, "contribution": round(0.15 * (1.0 if segment == "entrepreneur_actif" else 0.4), 4)}
            ]
            if bias_factor < 1.0:
                raw_score = sum(c["contribution"] for c in contributions)
                contributions.append({
                    "variable": "Pénalité disponibilité (T08)", 
                    "weight": "multiplicatif", 
                    "value": round(bias_factor, 4), 
                    "contribution": round(raw_score * bias_factor - raw_score, 4)
                })

            rankings.append({
                "learner_id": uid,
                "score": round(score, 4),
                "contributions": contributions,
                "metrics": {
                    "action_count": int(action_count),
                    "business_launched": bool(business_launched),
                    "avg_engagement": round(avg_engagement, 4),
                    "avg_completion": round(avg_completion, 4),
                    "categories_explored": int(categories_count),
                    "segment": segment
                }
            })
            
        # Trier par score décroissant
        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings[:top_n]
