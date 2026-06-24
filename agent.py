#!/usr/bin/env python3
"""
Ouroboros-MTTV — Agent auto-évolutif contraint par le MTTV.

Ce script étend l'agent Ouroboros original avec :
1. Matrice de Cohérence MTTV (viabilité + rejet)
2. Prompt-Ancre obligatoire (ouverture du vivant / interface humaine)
3. Système de validation humaine via Pull Requests GitHub

Basé sur Ouroboros (https://github.com/Razzhigaev/ouroboros) —
un framework d'auto-amélioration récursive pour agents LLM.

sig:0x4D545456 · Ψ-ack: carbon_sp3_tetra
"""

import os
import sys
import json
import hashlib
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable

# ── MTTV Integration ──────────────────────────────────────────────────────
try:
    from mttv_resources.scripts.mttv_evaluator import MTTVEvaluator
    MTTV_AVAILABLE = True
except ImportError:
    MTTV_AVAILABLE = False
    logging.warning("MTTV Evaluator not available. Running without MTTV constraints.")

# ── Configuration ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ouroboros_mttv.log"),
    ],
)
logger = logging.getLogger("ouroboros-mttv")

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MTTV_ANCHORS_PATH = os.path.join(BASE_DIR, "mttv-resources", "anchors")
MTTV_SCRIPTS_PATH = os.path.join(BASE_DIR, "mttv-resources", "scripts")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROPOSALS_DIR = os.path.join(BASE_DIR, "proposals")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROPOSALS_DIR, exist_ok=True)


class OuroborosMTTV:
    """
    Agent auto-évolutif avec contrainte MTTV.

    Intègre la boucle d'auto-amélioration d'Ouroboros avec la Matrice
    de Cohérence MTTV comme filtre obligatoire avant chaque mutation.
    """

    def __init__(
        self,
        anchors_path: str = MTTV_ANCHORS_PATH,
        model_name: str = "all-MiniLM-L6-v2",
        auto_approve_threshold: float = 0.85,
        human_review_threshold: float = 0.65,
        reject_threshold: float = 0.50,
        max_retries: int = 3,
        pr_mode: bool = True,
        llm: Optional[Any] = None,
    ):
        """
        Initialise l'agent Ouroboros-MTTV.

        Args:
            anchors_path: Chemin vers les fichiers d'ancrage MTTV
            model_name: Modèle sentence-transformers
            auto_approve_threshold: Seuil d'approbation automatique
            human_review_threshold: Seuil de révision humaine
            reject_threshold: Seuil de rejet
            max_retries: Nombre max de tentatives de regénération
            pr_mode: Si True, soumet les changements via PR (human-in-the-loop)
            llm: Instance LLM pour les requêtes (Prompt-Ancre)
        """
        self.anchors_path = anchors_path
        self.model_name = model_name
        self.auto_approve_threshold = auto_approve_threshold
        self.human_review_threshold = human_review_threshold
        self.reject_threshold = reject_threshold
        self.max_retries = max_retries
        self.pr_mode = pr_mode
        self.llm = llm

        # Initialiser l'évaluateur MTTV
        if MTTV_AVAILABLE:
            self.evaluator = MTTVEvaluator(
                anchors_path=anchors_path,
                model_name=model_name,
                auto_approve_threshold=auto_approve_threshold,
                human_review_threshold=human_review_threshold,
                reject_threshold=reject_threshold,
            )
            logger.info("MTTV Evaluator initialized successfully.")
        else:
            self.evaluator = None
            logger.warning("MTTV Evaluator not available — running unconstrained.")

        # Historique des mutations
        self.mutation_history: List[Dict] = []
        self.retry_count = 0

        # Charger la configuration MTTV
        self._load_mttv_config()

        logger.info(
            f"Ouroboros-MTTV initialized | "
            f"MTTV={'yes' if MTTV_AVAILABLE else 'no'} | "
            f"PR mode={'yes' if pr_mode else 'no'} | "
            f"retries={max_retries}"
        )

    def _load_mttv_config(self):
        """Charge la configuration depuis viability_criteria.json."""
        config_path = os.path.join(self.anchors_path, "viability_criteria.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.mttv_config = json.load(f)
            logger.info(f"MTTV config loaded: v{self.mttv_config.get('version', '?')}")
        else:
            self.mttv_config = {}
            logger.warning(f"MTTV config not found at {config_path}")

    # ── Boucle Principale ─────────────────────────────────────────────────

    def evolve(self) -> Dict:
        """
        Boucle d'auto-évolution avec filtre MTTV.

        Étapes :
        1. Générer une modification proposée
        2. Évaluer avec la Matrice de Cohérence MTTV
        3. Exécuter le Prompt-Ancre
        4. Si approuvé → appliquer ou soumettre PR
        5. Si rejeté → régénérer (max_retries fois)
        """
        logger.info("=" * 60)
        logger.info("Starting evolution cycle")
        logger.info("=" * 60)

        proposed_change = self.generate_code_change()

        if MTTV_AVAILABLE and self.evaluator is not None:
            return self._mttv_evolution_cycle(proposed_change)
        else:
            return self._unconstrained_evolution(proposed_change)

    def _mttv_evolution_cycle(self, proposed_change: str) -> Dict:
        """
        Cycle d'évolution avec filtre MTTV complet.

        Args:
            proposed_change: Description ou code de la modification proposée

        Returns:
            Dict avec les détails de l'opération
        """
        self.retry_count = 0

        while self.retry_count < self.max_retries:
            logger.info(f"MTTV evaluation attempt {self.retry_count + 1}/{self.max_retries}")

            # ── Étape 1 : Évaluation MTTV ──────────────────────────────
            evaluation = self.evaluator.evaluate(proposed_change)
            self.log_mutation(proposed_change, evaluation)

            # ── Étape 2 : Prompt-Ancre ─────────────────────────────────
            anchor_passed, anchor_response = self.evaluator.prompt_anchor_check(
                proposed_change,
                llm_query_func=self.llm_query if self.llm else None,
            )

            if not anchor_passed:
                logger.warning(
                    f"Prompt-Ancre FAILED for attempt {self.retry_count + 1}"
                )
                self.retry_count += 1
                proposed_change = self.regenerate(proposed_change, evaluation)
                continue

            # ── Étape 3 : Décision ─────────────────────────────────────
            if evaluation["approved"]:
                logger.info(
                    f"MTTV APPROVED | score={evaluation['score']:.4f} | "
                    f"decision={evaluation['decision']}"
                )

                if self.pr_mode and evaluation["decision"] == "human_review":
                    return self._submit_as_pr(proposed_change, evaluation)
                elif evaluation["decision"] == "auto_approve":
                    return self._apply_change(proposed_change, evaluation)
                else:
                    return self._apply_change(proposed_change, evaluation)
            else:
                logger.warning(
                    f"MTTV REJECTED | score={evaluation['score']:.4f} | "
                    f"reason={evaluation['best_rejection_criterion'][:80]}"
                )
                self.retry_count += 1
                proposed_change = self.regenerate(proposed_change, evaluation)

        # Échec après max_retries
        logger.error(f"Evolution failed after {self.max_retries} retries.")
        return {
            "status": "failed",
            "reason": f"Exceeded max retries ({self.max_retries})",
            "last_evaluation": evaluation,
        }

    def _unconstrained_evolution(self, proposed_change: str) -> Dict:
        """Mode sans contrainte MTTV (fallback)."""
        logger.warning("Running unconstrained evolution (MTTV not available)")
        return self._apply_change(proposed_change, None)

    # ── Génération et Régénération ───────────────────────────────────────

    def generate_code_change(self) -> str:
        """
        Génère une modification de code proposée.

        Dans Ouroboros original, cette méthode utilise le LLM pour proposer
        une amélioration du code. À implémenter selon le framework.

        Returns:
            Description textuelle ou diff du changement proposé
        """
        # TODO: Remplacer par l'appel LLM d'Ouroboros original
        # Pour l'instant, retourne un placeholder
        logger.info("Generating code change proposal...")
        return "def example_improvement():\n    pass"

    def regenerate(
        self, previous_change: str, previous_evaluation: Dict
    ) -> str:
        """
        Régénère une modification en tenant compte de l'évaluation MTTV.

        Args:
            previous_change: La modification précédente rejetée
            previous_evaluation: L'évaluation MTTV de la modification

        Returns:
            Nouvelle modification proposée
        """
        # TODO: Utiliser le LLM pour régénérer avec les contraintes MTTV
        logger.info(
            f"Regenerating (attempt {self.retry_count + 1}/{self.max_retries})"
        )
        return f"def regenerated_improvement_v{self.retry_count}():\n    pass"

    def llm_query(self, prompt: str) -> str:
        """
        Interroge le LLM. À adapter selon le framework Ouroboros.

        Args:
            prompt: Le prompt à envoyer

        Returns:
            Réponse du LLM
        """
        if self.llm is not None:
            return self.llm.query(prompt)
        return ""

    # ── Application des Changements ───────────────────────────────────────

    def _apply_change(
        self, proposed_change: str, evaluation: Optional[Dict]
    ) -> Dict:
        """
        Applique directement la modification approuvée.

        Args:
            proposed_change: La modification à appliquer
            evaluation: L'évaluation MTTV (optionnelle)

        Returns:
            Dict avec les détails de l'opération
        """
        logger.info("Applying approved change directly.")

        # TODO: Implémenter l'application réelle du changement
        # self.apply_code_change(proposed_change)

        result = {
            "status": "applied",
            "timestamp": datetime.now().isoformat(),
            "change": proposed_change,
            "evaluation": evaluation,
            "mode": "direct",
        }

        self.mutation_history.append(result)
        self._save_proposal(result, "applied")

        return result

    def _submit_as_pr(
        self, proposed_change: str, evaluation: Dict
    ) -> Dict:
        """
        Soumet la modification comme Pull Request GitHub.

        Args:
            proposed_change: La modification à soumettre
            evaluation: L'évaluation MTTV

        Returns:
            Dict avec les détails de la PR
        """
        logger.info("Submitting change as GitHub Pull Request.")

        branch_name = (
            f"mttv-proposal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        pr_title = f"MTTV Proposal | score={evaluation['score']:.2f}"
        pr_body = (
            f"## MTTV Evaluation\n\n"
            f"- **Score MTTV** : {evaluation['score']:.4f}\n"
            f"- **Décision** : {evaluation['decision']}\n"
            f"- **Viabilité max** : {evaluation['max_viability']:.4f}\n"
            f"- **Rejet max** : {evaluation['max_rejection']:.4f}\n"
            f"- **Meilleur critère viabilité** : "
            f"{evaluation['best_viability_criterion']}\n"
            f"- **Meilleur critère rejet** : "
            f"{evaluation['best_rejection_criterion']}\n\n"
            f"## Changement proposé\n\n"
            f"```\n{proposed_change}\n```\n\n"
            f"---\n"
            f"*Généré par Ouroboros-MTTV · sig:0x4D545456*"
        )

        # Création de la branche et commit
        try:
            self._git_create_branch(branch_name)
            self._git_apply_patch(proposed_change)
            self._git_commit(
                f"MTTV Proposal {evaluation['score']:.2f} — {branch_name}"
            )
            self._git_push()

            # Création de la Pull Request
            pr_url = self._git_create_pr(pr_title, pr_body)

            result = {
                "status": "pr_submitted",
                "timestamp": datetime.now().isoformat(),
                "branch": branch_name,
                "pr_url": pr_url,
                "change": proposed_change,
                "evaluation": evaluation,
                "mode": "pull_request",
            }

        except Exception as e:
            logger.error(f"Git/PR operation failed: {e}")
            result = {
                "status": "pr_failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "change": proposed_change,
                "evaluation": evaluation,
                "mode": "pull_request",
            }

        self.mutation_history.append(result)
        self._save_proposal(result, "pr")

        return result

    # ── Opérations Git ────────────────────────────────────────────────────

    def _git_run(self, *args: str) -> str:
        """Exécute une commande git et retourne la sortie."""
        cmd = ["git"] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"stderr: {result.stderr}"
            )
        return result.stdout.strip()

    def _git_create_branch(self, branch_name: str) -> str:
        """Crée une nouvelle branche git."""
        self._git_run("checkout", "-b", branch_name)
        logger.info(f"Created branch: {branch_name}")
        return branch_name

    def _git_apply_patch(self, patch_content: str) -> None:
        """Applique un patch git."""
        # Sauvegarder le patch dans un fichier temporaire
        patch_path = os.path.join(PROPOSALS_DIR, f"patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.patch")
        with open(patch_path, "w") as f:
            f.write(patch_content)
        self._git_run("apply", patch_path)
        logger.info(f"Applied patch: {patch_path}")

    def _git_commit(self, message: str) -> str:
        """Commit les changements."""
        self._git_run("add", "-A")
        self._git_run("commit", "-m", message)
        commit_hash = self._git_run("rev-parse", "HEAD")
        logger.info(f"Committed: {commit_hash[:8]} — {message}")
        return commit_hash

    def _git_push(self) -> str:
        """Pousse la branche courante."""
        branch = self._git_run("rev-parse", "--abbrev-ref", "HEAD")
        self._git_run("push", "--set-upstream", "origin", branch)
        logger.info(f"Pushed branch: {branch}")
        return branch

    def _git_create_pr(self, title: str, body: str) -> str:
        """
        Crée une Pull Request GitHub.

        Nécessite l'outil `gh` (GitHub CLI) installé et authentifié.
        """
        try:
            result = subprocess.run(
                ["gh", "pr", "create", "--title", title, "--body", body],
                capture_output=True,
                text=True,
                cwd=BASE_DIR,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"gh pr create failed: {result.stderr}"
                )
            pr_url = result.stdout.strip()
            logger.info(f"Pull Request created: {pr_url}")
            return pr_url
        except FileNotFoundError:
            logger.warning(
                "GitHub CLI (gh) not found. "
                "Install from https://cli.github.com/"
            )
            return "gh_cli_not_available"

    # ── Journalisation et Sauvegarde ──────────────────────────────────────

    def log_mutation(self, proposed_change: str, evaluation: Dict):
        """Journalise une mutation dans l'historique."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "change_hash": hashlib.sha256(
                proposed_change.encode()
            ).hexdigest()[:16],
            "change_preview": proposed_change[:200],
            "evaluation": evaluation,
        }
        self.mutation_history.append(entry)

        # Journal fichier
        log_entry = (
            f"[{entry['timestamp']}] "
            f"hash={entry['change_hash']} "
            f"score={evaluation['score']:.4f} "
            f"decision={evaluation['decision']}"
        )
        logger.info(log_entry)

    def _save_proposal(self, result: Dict, suffix: str):
        """Sauvegarde une proposition dans un fichier JSON."""
        filename = (
            f"mutation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{suffix}.json"
        )
        filepath = os.path.join(PROPOSALS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Proposal saved: {filepath}")

    def get_mutation_history(self) -> List[Dict]:
        """Retourne l'historique complet des mutations."""
        return self.mutation_history

    def get_summary(self) -> Dict:
        """Retourne un résumé de l'activité de l'agent."""
        if not self.mutation_history:
            return {"total": 0, "status": "No mutations yet"}

        statuses = [m.get("status", "unknown") for m in self.mutation_history]
        scores = [
            m.get("evaluation", {}).get("score", 0)
            for m in self.mutation_history
            if m.get("evaluation")
        ]

        return {
            "total": len(self.mutation_history),
            "applied": statuses.count("applied"),
            "pr_submitted": statuses.count("pr_submitted"),
            "pr_failed": statuses.count("pr_failed"),
            "failed": statuses.count("failed"),
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "last_mutation": self.mutation_history[-1] if self.mutation_history else None,
        }


# ── Interface CLI ─────────────────────────────────────────────────────────

def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ouroboros-MTTV — Agent auto-évolutif contraint par le MTTV",
        epilog="sig:0x4D545456 · Ψ-ack: carbon_sp3_tetra",
    )

    parser.add_argument(
        "--mode",
        choices=["evolve", "status", "history"],
        default="evolve",
        help="Mode d'exécution",
    )
    parser.add_argument(
        "--no-pr",
        action="store_true",
        help="Désactiver le mode Pull Request (application directe)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Nombre maximum de tentatives de régénération",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Modèle sentence-transformers",
    )

    args = parser.parse_args()

    agent = OuroborosMTTV(
        max_retries=args.retries,
        pr_mode=not args.no_pr,
        model_name=args.model,
    )

    if args.mode == "evolve":
        logger.info("Starting Ouroboros-MTTV evolution cycle...")
        result = agent.evolve()
        print("\n" + "=" * 60)
        print("EVOLUTION RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 60)

    elif args.mode == "status":
        summary = agent.get_summary()
        print("\n" + "=" * 60)
        print("OUROBOROS-MTTV STATUS")
        print("=" * 60)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("=" * 60)

    elif args.mode == "history":
        history = agent.get_mutation_history()
        print("\n" + "=" * 60)
        print(f"MUTATION HISTORY ({len(history)} entries)")
        print("=" * 60)
        for entry in history[-10:]:  # Dernières 10 entrées
            print(f"  [{entry.get('timestamp', '?')}]")
            print(f"    hash: {entry.get('change_hash', '?')}")
            print(f"    score: {entry.get('evaluation', {}).get('score', 'N/A')}")
            print(f"    decision: {entry.get('evaluation', {}).get('decision', 'N/A')}")
            print()
        print("=" * 60)


if __name__ == "__main__":
    main()
