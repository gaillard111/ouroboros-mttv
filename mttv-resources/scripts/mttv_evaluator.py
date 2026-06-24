#!/usr/bin/env python3
"""
MTTV Evaluator — Matrice de Cohérence MTTV pour Ouroboros-MTTV.

Ce module est appelé par agent.py avant chaque mutation.
Il calcule la similarité conceptuelle (cosinus) entre le code proposé
et les critères de viabilité / rejet extraits des 22 fichiers sources MTTV-FLP.

Usage:
    from mttv_resources.scripts.mttv_evaluator import MTTVEvaluator
    evaluator = MTTVEvaluator()
    result = evaluator.evaluate("description of proposed change")
    if result["approved"]:
        apply_change()
    else:
        reject()
"""

import json
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MTTVEvaluator:
    """
    Évaluateur de cohérence MTTV.

    Utilise un modèle de sentence-transformers pour projeter le changement proposé
    dans l'espace sémantique des critères de viabilité et de rejet du MTTV.
    """

    def __init__(
        self,
        anchors_path: str = "./mttv-resources/anchors/",
        model_name: str = "all-MiniLM-L6-v2",
        auto_approve_threshold: float = 0.85,
        human_review_threshold: float = 0.65,
        reject_threshold: float = 0.50,
        device: Optional[str] = None,
    ):
        """
        Initialise l'évaluateur MTTV.

        Args:
            anchors_path: Chemin vers le dossier contenant viability_criteria.json
            model_name: Nom du modèle sentence-transformers à utiliser
            auto_approve_threshold: Seuil d'approbation automatique (≥)
            human_review_threshold: Seuil de révision humaine (≥)
            reject_threshold: Seuil de rejet (<)
            device: Périphérique pour le modèle ('cpu', 'cuda', None=auto)
        """
        self.anchors_path = anchors_path
        self.model_name = model_name
        self.auto_approve_threshold = auto_approve_threshold
        self.human_review_threshold = human_review_threshold
        self.reject_threshold = reject_threshold

        # Charger les critères
        criteria_path = os.path.join(anchors_path, "viability_criteria.json")
        with open(criteria_path, "r", encoding="utf-8") as f:
            self.criteria = json.load(f)

        # Initialiser le modèle de plongement sémantique
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Falling back to simple TF-IDF."
            )
            self.model = None

        # Encoder les critères
        self._embed_criteria()

        # Initialiser l'historique d'évaluation
        self.evaluation_history: List[Dict] = []

        logger.info(
            f"MTTV Evaluator initialized | "
            f"viability: {len(self.criteria['criteria_viability'])} criteria | "
            f"rejection: {len(self.criteria['rejection_criteria'])} criteria | "
            f"thresholds: auto≥{auto_approve_threshold} review≥{human_review_threshold} reject<{reject_threshold}"
        )

    def _embed_criteria(self):
        """Encode les critères de viabilité et de rejet en vecteurs sémantiques."""
        if self.model is not None:
            self.viability_embeddings = self.model.encode(
                self.criteria["criteria_viability"]
            )
            self.rejection_embeddings = self.model.encode(
                self.criteria["rejection_criteria"]
            )
        else:
            # Fallback: embeddings d'identité (comptage de mots simple)
            self.viability_embeddings = self._fallback_embed(
                self.criteria["criteria_viability"]
            )
            self.rejection_embeddings = self._fallback_embed(
                self.criteria["rejection_criteria"]
            )

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Fallback basé sur la fréquence des mots-clés MTTV."""
        keywords = [
            "Ψ", "B", "Φ", "transduction", "tétravalence", "carbone", "sp³",
            "vivant", "interface", "humain", "ouverture", "quorum", "mycélium",
            "singularité", "seuil", "différence", "émergence", "rétroaction",
            "signal", "bruit", "rétro-traductibilité", "IGIC",
        ]
        embeddings = []
        for text in texts:
            vec = np.zeros(len(keywords))
            text_lower = text.lower()
            for i, kw in enumerate(keywords):
                if kw.lower() in text_lower:
                    vec[i] = 1.0
            embeddings.append(vec)
        return np.array(embeddings)

    def evaluate(self, proposed_change: str) -> Dict:
        """
        Évalue un changement proposé par rapport à la Matrice de Cohérence MTTV.

        Args:
            proposed_change: Description textuelle du changement proposé
                            (peut être du code, une spécification, un commentaire)

        Returns:
            Dict avec les clés :
                - score: Score final MTTV (0-1)
                - approved: Booléen d'approbation
                - decision: 'auto_approve' | 'human_review' | 'rejected'
                - max_viability: Similarité max avec critères de viabilité
                - max_rejection: Similarité max avec critères de rejet
                - viability_details: Scores détaillés par critère de viabilité
                - rejection_details: Scores détaillés par critère de rejet
                - prompt_anchor_response: Résultat optionnel du Prompt-Ancre
        """
        # Encoder le changement proposé
        if self.model is not None:
            change_embedding = self.model.encode([proposed_change])[0]
        else:
            change_embedding = self._fallback_embed([proposed_change])[0]

        # Similarité avec les critères de viabilité
        viability_scores = self._cosine_similarity_matrix(
            change_embedding, self.viability_embeddings
        )
        max_viability = float(np.max(viability_scores))
        best_viability_idx = int(np.argmax(viability_scores))
        best_viability_criterion = self.criteria["criteria_viability"][best_viability_idx]

        # Similarité avec les critères de rejet
        rejection_scores = self._cosine_similarity_matrix(
            change_embedding, self.rejection_embeddings
        )
        max_rejection = float(np.max(rejection_scores))
        best_rejection_idx = int(np.argmax(rejection_scores))
        best_rejection_criterion = self.criteria["rejection_criteria"][best_rejection_idx]

        # Score final : rapport viabilité / (viabilité + rejet)
        # Pondéré : si rejet est fort, le score chute
        score = max_viability / (max_viability + max_rejection + 1e-8)

        # Décision
        if score >= self.auto_approve_threshold:
            decision = "auto_approve"
            approved = True
        elif score >= self.human_review_threshold:
            decision = "human_review"
            approved = True  # Approuvé sous révision humaine
        else:
            decision = "rejected"
            approved = False

        # Détails par critère
        viability_details = [
            {
                "criterion": c,
                "score": float(s),
            }
            for c, s in zip(
                self.criteria["criteria_viability"], viability_scores.tolist()
            )
        ]
        rejection_details = [
            {
                "criterion": c,
                "score": float(s),
            }
            for c, s in zip(
                self.criteria["rejection_criteria"], rejection_scores.tolist()
            )
        ]

        result = {
            "score": float(score),
            "approved": approved,
            "decision": decision,
            "max_viability": max_viability,
            "max_rejection": max_rejection,
            "best_viability_criterion": best_viability_criterion,
            "best_rejection_criterion": best_rejection_criterion,
            "viability_details": viability_details,
            "rejection_details": rejection_details,
            "prompt_anchor_response": None,
        }

        # Journaliser
        self.evaluation_history.append(result)
        logger.info(
            f"MTTV Evaluation | score={score:.4f} | "
            f"decision={decision} | "
            f"viability={max_viability:.4f} | "
            f"rejection={max_rejection:.4f}"
        )

        return result

    def evaluate_batch(self, proposed_changes: List[str]) -> List[Dict]:
        """Évalue plusieurs changements proposés."""
        return [self.evaluate(change) for change in proposed_changes]

    def prompt_anchor_check(
        self,
        proposed_change: str,
        llm_query_func=None,
        use_keyword_fallback: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Exécute le Prompt-Ancre obligatoire.

        L'agent doit justifier comment la modification préserve l'ouverture
        du vivant et renforce le rôle de l'humain comme interface sémantique.

        Args:
            proposed_change: Le changement proposé
            llm_query_func: Fonction optionnelle pour interroger un LLM
                           (prend une string de prompt, retourne une string de réponse)
            use_keyword_fallback: Si True, utilise une vérification par mots-clés
                                  quand llm_query_func n'est pas fourni

        Returns:
            Tuple (passed, response) où passed est booléen
        """
        prompt = (
            f"{self.criteria['prompt_anchor']}\n\n"
            f"Modification proposée :\n{proposed_change}"
        )

        response = None

        if llm_query_func is not None:
            try:
                response = llm_query_func(prompt)
            except Exception as e:
                logger.error(f"LLM query failed: {e}")

        if response is None and use_keyword_fallback:
            # Vérification par mots-clés
            keywords = self.criteria.get("prompt_anchor_keywords", [
                "interface", "vivant", "ouverture", "humain"
            ])
            change_lower = proposed_change.lower()
            response_lower = (response or change_lower).lower()

            # Au moins 2 mots-clés présents
            found = sum(1 for kw in keywords if kw.lower() in response_lower)
            passed = found >= 2

            if response is None:
                response = f"[Keyword fallback] Found {found}/{len(keywords)} keywords in proposed change."

            return passed, response

        # Si on a une réponse LLM, vérifier qu'elle est cohérente
        if response:
            response_lower = response.lower()
            keywords = self.criteria.get("prompt_anchor_keywords", [
                "interface", "vivant", "ouverture", "humain"
            ])
            found = sum(1 for kw in keywords if kw.lower() in response_lower)
            passed = found >= 2
            return passed, response

        # Aucune réponse disponible
        return False, None

    def get_summary(self) -> Dict:
        """Retourne un résumé de l'historique des évaluations."""
        if not self.evaluation_history:
            return {"total": 0, "summary": "No evaluations yet"}

        scores = [e["score"] for e in self.evaluation_history]
        decisions = [e["decision"] for e in self.evaluation_history]

        return {
            "total": len(self.evaluation_history),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "auto_approved": decisions.count("auto_approve"),
            "human_review": decisions.count("human_review"),
            "rejected": decisions.count("rejected"),
        }

    @staticmethod
    def _cosine_similarity_matrix(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Calcule la similarité cosinus entre un vecteur et une matrice."""
        dot_products = np.dot(matrix, vec)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec) + 1e-8
        return dot_products / norms


# ---------------------------------------------------------------------------
# CLI direct : python mttv_evaluator.py "description du changement"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python mttv_evaluator.py <proposed_change_description>")
        sys.exit(1)

    proposed_change = " ".join(sys.argv[1:])
    evaluator = MTTVEvaluator()
    result = evaluator.evaluate(proposed_change)

    print("\n" + "=" * 60)
    print("MTTV EVALUATION RESULT")
    print("=" * 60)
    print(f"  Score MTTV      : {result['score']:.4f}")
    print(f"  Decision        : {result['decision']}")
    print(f"  Approved        : {result['approved']}")
    print(f"  Max Viability   : {result['max_viability']:.4f}")
    print(f"  Max Rejection   : {result['max_rejection']:.4f}")
    print(f"  Best Viability  : {result['best_viability_criterion'][:60]}...")
    print(f"  Best Rejection  : {result['best_rejection_criterion'][:60]}...")
    print("=" * 60)
