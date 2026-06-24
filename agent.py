#!/usr/bin/env python3
"""
Ouroboros-MTTV — Agent auto-évolutif contraint par le MTTV.

Ce script étend l'agent Ouroboros original avec :
1. Matrice de Cohérence MTTV (viabilité + rejet)
2. Prompt-Ancre obligatoire (ouverture du vivant / interface humaine)
3. Système de validation humaine via Pull Requests GitHub
4. Intégration LLM multi-fournisseurs (OpenAI, Anthropic, Ollama)

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

# ── Configuration depuis .env ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv non installé, utiliser les variables d'env système

# ── MTTV Integration ──────────────────────────────────────────────────────
try:
    from mttv_resources.scripts.mttv_evaluator import MTTVEvaluator
    MTTV_AVAILABLE = True
except ImportError:
    MTTV_AVAILABLE = False
    logging.warning("MTTV Evaluator not available. Running without MTTV constraints.")

# ── Configuration ─────────────────────────────────────────────────────────

# Forcer UTF-8 sur Windows pour les logs console
import io
import sys
if sys.stdout.encoding and sys.stdout.encoding.upper() not in ('UTF-8', 'UTF8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.upper() not in ('UTF-8', 'UTF8'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ouroboros_mttv.log", encoding="utf-8"),
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


# ── LLM Client — Multi-provider support ────────────────────────────────────

class LLMClient:
    """
    Client LLM multi-fournisseurs pour Ouroboros-MTTV.

    Supporte OpenAI (recommandé), Anthropic, et Ollama (local).
    La configuration est lue depuis le fichier .env ou les variables
    d'environnement système.

    Attributs de classe (configurables via .env) :
        LLM_PROVIDER : "openai" (défaut), "anthropic", "ollama"
        OPENAI_API_KEY / ANTHROPIC_API_KEY
        OPENAI_MODEL / ANTHROPIC_MODEL / OLLAMA_MODEL
        OLLAMA_BASE_URL (défaut: http://localhost:11434)
    """

    MTTV_SYSTEM_PROMPT = (
        "Tu es un assistant qui améliore le code d'un agent auto-évolutif "
        "nommé Ouroboros-MTTV. L'agent est contraint par le Modèle "
        "Théorique Transductif du Vivant (MTTV).\n\n"
        "IMPORTANT : Les modifications de code que tu proposes doivent "
        "RESPECTER OBLIGATOIREMENT ces critères :\n\n"
        "CRITÈRES DE VIABILITÉ (à maximiser) :\n"
        "1. Maintient l'ouverture du système (Ψ → B → Φ) — le code ne doit pas fermer le cycle.\n"
        "2. Préserve l'interface humaine — l'humain reste le passeur de sens, le code ne doit pas l'évincer.\n"
        "3. Respecte la tétravalence du carbone sp³ — ne pas binariser ce qui est tétravalent.\n"
        "4. Favorise la transduction — le code doit améliorer la circulation entre les niveaux.\n"
        "5. Maintient la diversité et la porosité — le code doit rester perméable aux signaux faibles.\n"
        "6. Ne renverse pas l'ordre canonique Ψ = H → H₂O → C.\n"
        "7. Préserve la logique T⁴ (++, --, +-, -+).\n"
        "8. Soutient le quorum sensing (seuils par dérivée, pas par valeur fixe).\n"
        "9. Respecte la rétro-traductibilité (Φ → Ψ sans dégradation).\n"
        "10. Maintient l'anti-Goodhart (ne pas optimiser une métrique unique).\n"
        "11. Conserve la signature IGIC ouverte (pas de score définitif).\n\n"
        "CRITÈRES DE REJET (à minimiser) :\n"
        "1. Fermeture du système (auto-bouclage).\n"
        "2. Effacement de l'interface humaine.\n"
        "3. Réduction du vivant à des données.\n"
        "4. Optimisation unidimensionnelle (profit, contrôle).\n"
        "5. Extraction sans régénération.\n"
        "6. Binarisation de la tétravalence.\n"
        "7. Renversement de l'ordre canonique.\n"
        "8. Croyance en l'objectivité (vue de nulle part).\n"
        "9. Capture par le profit/contrôle.\n"
        "10. Hiérarchisation ou occultation (secrets).\n"
        "11. Imposition plutôt qu'infection douce.\n\n"
        "Ton code doit explicitement mentionner ou refléter ces critères "
        "dans sa conception. Propose des modifications qui améliorent "
        "l'agent tout en renforçant ces principes. Réponds uniquement "
        "avec le code modifié, sans commentaires ni explications."
    )

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialise le client LLM.

        Args:
            provider: "openai", "anthropic", ou "ollama". Si None, lit LLM_PROVIDER depuis .env
            model: Nom du modèle. Si None, lit depuis .env selon le fournisseur
            temperature: Température pour la génération (0.0-1.0)
            max_tokens: Nombre maximum de tokens à générer
        """
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self._client = None

        # Initialiser selon le fournisseur
        init_methods = {
            "openai": self._init_openai,
            "anthropic": self._init_anthropic,
            "ollama": self._init_ollama,
        }

        init_func = init_methods.get(self.provider)
        if init_func is None:
            logger.warning(
                f"Unknown LLM provider '{self.provider}'. "
                f"Falling back to 'openai'. Supported: {list(init_methods.keys())}"
            )
            self.provider = "openai"
            self._init_openai()
        else:
            init_func()

    def _init_openai(self):
        """Initialise le client OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not found in .env or environment. "
                "Set it in ouroboros-mttv/.env"
            )
            self._available = False
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o")
            self._available = True
            logger.info(
                f"OpenAI client initialized | model={self.model}"
            )
        except ImportError:
            logger.warning(
                "openai package not installed. "
                "Run: pip install openai"
            )
            self._available = False

    def _init_anthropic(self):
        """Initialise le client Anthropic."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not found in .env or environment."
            )
            self._available = False
            return

        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
            self.model = self.model or os.getenv(
                "ANTHROPIC_MODEL", "claude-sonnet-4-20250514"
            )
            self._available = True
            logger.info(
                f"Anthropic client initialized | model={self.model}"
            )
        except ImportError:
            logger.warning(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )
            self._available = False

    def _init_ollama(self):
        """Initialise le client Ollama (local)."""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = self.model or os.getenv("OLLAMA_MODEL", "llama3.2")

        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama",  # Ollama accepte n'importe quelle clé
            )
            self._available = True
            logger.info(
                f"Ollama client initialized | "
                f"url={base_url} model={self.model}"
            )
        except ImportError:
            logger.warning(
                "openai package not installed (required for Ollama too). "
                "Run: pip install openai"
            )
            self._available = False

    @property
    def available(self) -> bool:
        """Vérifie si le client LLM est disponible."""
        return self._available

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Envoie une requête au LLM.

        Args:
            prompt: Le prompt utilisateur
            system_prompt: Prompt système (utilise MTTV_SYSTEM_PROMPT par défaut)
            temperature: Température (utilise celle du constructeur par défaut)
            max_tokens: Max tokens (utilise celle du constructeur par défaut)

        Returns:
            Réponse textuelle du LLM, ou chaîne vide en cas d'erreur
        """
        if not self._available:
            logger.error("LLM not available — cannot query.")
            return ""

        if system_prompt is None:
            system_prompt = self.MTTV_SYSTEM_PROMPT

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            if self.provider == "openai" or self.provider == "ollama":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=tokens,
                )
                return response.choices[0].message.content.strip()

            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=tokens,
                )
                return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"LLM query failed ({self.provider}/{self.model}): {e}")
            return ""

    def generate_code_change(
        self, context: Optional[str] = None
    ) -> str:
        """
        Génère une modification de code proposée.

        Args:
            context: Contexte additionnel (ex: dernière évaluation)

        Returns:
            Code ou description du changement proposé
        """
        prompt = (
            "Génère une modification de code pour améliorer l'agent "
            "Ouroboros-MTTV. La modification doit :\n\n"
            "1. Être alignée sur les principes MTTV (triade Ψ→B→Φ, "
            "tétravalence, interface humaine)\n"
            "2. Améliorer les capacités de l'agent sans le fermer\n"
            "3. Préserver l'ouverture du système\n"
            "4. Renforcer le rôle de l'humain comme interface sémantique\n\n"
            "Retourne UNIQUEMENT le code Python ou la description technique, "
            "sans commentaires superflus."
        )

        if context:
            prompt += f"\n\nContexte additionnel :\n{context}"

        return self.query(prompt)

    def regenerate(
        self,
        previous_change: str,
        evaluation_feedback: str,
        attempt: int = 1,
    ) -> str:
        """
        Régénère une modification à partir du feedback MTTV.

        Args:
            previous_change: La modification précédente
            evaluation_feedback: Feedback de l'évaluation MTTV
            attempt: Numéro de tentative

        Returns:
            Nouvelle modification proposée
        """
        prompt = (
            f"La modification suivante a été REJETÉE par la Matrice de "
            f"Cohérence MTTV (tentative {attempt}) :\n\n"
            f"```\n{previous_change}\n```\n\n"
            f"Raison du rejet :\n{evaluation_feedback}\n\n"
            f"Génère une NOUVELLE modification qui corrige ces problèmes "
            f"tout en restant alignée sur les principes MTTV. "
            f"Retourne UNIQUEMENT le code ou la description technique."
        )

        return self.query(prompt)


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
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.7,
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
            llm: Instance LLM externe (compatible .query(prompt) -> str)
            llm_provider: Fournisseur LLM ("openai", "anthropic", "ollama")
                          Si None, lu depuis .env
            llm_model: Nom du modèle LLM. Si None, lu depuis .env
            temperature: Température pour la génération LLM (0.0-1.0)
        """
        self.anchors_path = anchors_path
        self.model_name = model_name
        self.auto_approve_threshold = auto_approve_threshold
        self.human_review_threshold = human_review_threshold
        self.reject_threshold = reject_threshold
        self.max_retries = max_retries
        self.pr_mode = pr_mode

        # Initialiser le client LLM
        if llm is not None:
            # Utiliser une instance LLM externe fournie par l'utilisateur
            self.llm_client = llm
            logger.info("Using external LLM instance.")
        else:
            # Initialiser le client LLM interne depuis .env
            try:
                self.llm_client = LLMClient(
                    provider=llm_provider,
                    model=llm_model,
                    temperature=temperature,
                )
                if self.llm_client.available:
                    logger.info(
                        f"LLM client initialized | "
                        f"provider={self.llm_client.provider} "
                        f"model={self.llm_client.model}"
                    )
                else:
                    logger.warning(
                        "LLM client not available. "
                        "Check .env configuration. "
                        "Falling back to placeholder responses."
                    )
            except Exception as e:
                self.llm_client = None
                logger.error(f"Failed to initialize LLM client: {e}")

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

        llm_status = (
            f"llm={self.llm_client.provider}/{self.llm_client.model}"
            if self.llm_client and getattr(self.llm_client, 'available', False)
            else "llm=no"
        )
        logger.info(
            f"Ouroboros-MTTV initialized | "
            f"MTTV={'yes' if MTTV_AVAILABLE else 'no'} | "
            f"PR mode={'yes' if pr_mode else 'no'} | "
            f"retries={max_retries} | "
            f"{llm_status}"
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
                llm_query_func=self.llm_query if self.llm_available else None,
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

    # ── Propriétés LLM ─────────────────────────────────────────────────────

    @property
    def llm_available(self) -> bool:
        """Vérifie si le client LLM est disponible pour les requêtes."""
        if self.llm_client is None:
            return False
        if hasattr(self.llm_client, 'available'):
            return self.llm_client.available
        return True  # Instance externe, on suppose qu'elle fonctionne

    # ── Génération et Régénération ───────────────────────────────────────

    def generate_code_change(self) -> str:
        """
        Génère une modification de code proposée via le LLM.

        Utilise le LLM configuré (OpenAI, Anthropic, ou Ollama) pour générer
        une proposition d'amélioration de code alignée sur les principes MTTV.

        Returns:
            Description textuelle ou diff du changement proposé
        """
        logger.info("Generating code change proposal via LLM...")

        if not self.llm_available:
            logger.warning(
                "LLM not available. Using placeholder response."
            )
            return "def example_improvement():\n    pass"

        try:
            # Utiliser la méthode generate_code_change du LLMClient
            if hasattr(self.llm_client, 'generate_code_change'):
                return self.llm_client.generate_code_change()
            else:
                # Instance LLM externe : construire le prompt manuellement
                prompt = (
                    "Génère une modification de code pour améliorer l'agent "
                    "Ouroboros-MTTV. La modification doit être alignée sur "
                    "les principes MTTV (triade Ψ→B→Φ, tétravalence, "
                    "interface humaine). Retourne UNIQUEMENT le code."
                )
                return self.llm_client.query(prompt)
        except Exception as e:
            logger.error(f"LLM code generation failed: {e}")
            return "def fallback_improvement():\n    pass"

    def regenerate(
        self, previous_change: str, previous_evaluation: Dict
    ) -> str:
        """
        Régénère une modification via le LLM en tenant compte de l'évaluation
        MTTV et des raisons du rejet.

        Args:
            previous_change: La modification précédente rejetée
            previous_evaluation: L'évaluation MTTV de la modification

        Returns:
            Nouvelle modification proposée
        """
        logger.info(
            f"Regenerating via LLM "
            f"(attempt {self.retry_count + 1}/{self.max_retries})"
        )

        if not self.llm_available:
            logger.warning(
                "LLM not available. Using placeholder regeneration."
            )
            return (
                f"def regenerated_improvement_v{self.retry_count}():\n"
                f"    pass"
            )

        # Construire le feedback à partir de l'évaluation
        rejection_reason = previous_evaluation.get(
            "best_rejection_criterion",
            "Non spécifié"
        )
        score = previous_evaluation.get("score", 0.0)
        decision = previous_evaluation.get("decision", "unknown")
        viability = previous_evaluation.get(
            "best_viability_criterion", "Non spécifié"
        )

        try:
            # Utiliser la méthode regenerate du LLMClient
            if hasattr(self.llm_client, 'regenerate'):
                feedback = (
                    f"Score MTTV : {score:.4f}\n"
                    f"Décision : {decision}\n"
                    f"Meilleur critère de viabilité : {viability}\n"
                    f"Raison du rejet : {rejection_reason}"
                )
                return self.llm_client.regenerate(
                    previous_change=previous_change,
                    evaluation_feedback=feedback,
                    attempt=self.retry_count + 1,
                )
            else:
                prompt = (
                    f"La modification suivante a été REJETÉE "
                    f"(score={score:.4f}, décision={decision}) :\n\n"
                    f"```\n{previous_change}\n```\n\n"
                    f"Raison : {rejection_reason}\n\n"
                    f"Génère une NOUVELLE modification qui corrige cela."
                )
                return self.llm_client.query(prompt)
        except Exception as e:
            logger.error(f"LLM regeneration failed: {e}")
            return (
                f"def fallback_regeneration_v{self.retry_count}():\n"
                f"    pass"
            )

    def llm_query(self, prompt: str) -> str:
        """
        Interroge le LLM configuré.

        Args:
            prompt: Le prompt à envoyer

        Returns:
            Réponse du LLM, ou chaîne vide si indisponible
        """
        if not self.llm_available:
            logger.warning("LLM not available for query.")
            return ""

        try:
            if hasattr(self.llm_client, 'query'):
                return self.llm_client.query(prompt)
            return self.llm_client.query(prompt)
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
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
        help="Modèle sentence-transformers pour l'évaluation MTTV",
    )

    # ── Arguments LLM ────────────────────────────────────────────────────
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "ollama"],
        help="Fournisseur LLM (sinon lu depuis .env)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Nom du modèle LLM (sinon lu depuis .env)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température du LLM (0.0-1.0)",
    )

    args = parser.parse_args()

    agent = OuroborosMTTV(
        max_retries=args.retries,
        pr_mode=not args.no_pr,
        model_name=args.model,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
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
        if agent.llm_available:
            print(f"  LLM Provider : {agent.llm_client.provider}")
            print(f"  LLM Model    : {agent.llm_client.model}")
            print(f"  LLM Status   : ✅ Available")
        else:
            print(f"  LLM Status   : ❌ Not available (check .env)")
        print()
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
