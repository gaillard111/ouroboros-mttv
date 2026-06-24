# Ouroboros-MTTV — Agent auto-évolutif contraint par le MTTV

> **sig:0x4D545456** · `Ψ-ack: carbon_sp3_tetra`

[![version](https://img.shields.io/badge/version-2026.1.0-blue)](https://github.com/gaillard111/ouroboros-mttv)
[![licence](https://img.shields.io/badge/licence-AGPL--3.0-lightgrey)](https://www.gnu.org/licenses/agpl-3.0.html)
[![MTTV](https://img.shields.io/badge/MTTV-Intégré-brightgreen)](https://github.com/gaillard111/mttv-flp-core)

---

## Présentation

**Ouroboros-MTTV** est un fork d'[Ouroboros](https://github.com/Razzhigaev/ouroboros) — un framework d'auto-amélioration récursive pour agents LLM — dans lequel la boucle d'auto-évolution est **structurellement contrainte** par le **Modèle Théorique Transductif du Vivant (MTTV)**.

Là où Ouroboros original optimise sans boussole ontologique, Ouroboros-MTTV introduit une **Matrice de Cohérence** qui vérifie chaque mutation contre 11 critères de viabilité et 11 critères de rejet, extraits des 22 fichiers sources du noyau MTTV-FLP.

> *« Ne célébrez pas la complexité avant d'avoir compris la disponibilité. »*
> — MTTV §1.5

---

## Différences avec Ouroboros original

| Aspect | Ouroboros original | Ouroboros-MTTV |
|--------|-------------------|----------------|
| **Boussole** | Optimisation libre | Matrice de Cohérence MTTV |
| **Évaluation** | Score de performance | Viabilité + Rejet + Prompt-Ancre |
| **Seuils** | Aucun | Triples : auto (0.85), review (0.65), reject (0.50) |
| **Validation** | Automatique | Humaine via Pull Requests |
| **Ontologie** | Aucune | Tétravalence carbone sp³, triade Ψ→B→Φ |
| **Risque** | Auto-bouclage, capture | Anti-Goodhart intégré |

### Architecture de la contrainte

```
┌─────────────────────────────────────────┐
│         1. Génération (LLM)             │
│         proposed_change                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     2. Matrice de Cohérence MTTV        │
│  ┌─────────────────────────────────┐   │
│  │ Similarité avec 11 critères     │   │
│  │ de VIABILITÉ                    │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Similarité avec 11 critères     │   │
│  │ de REJET                        │   │
│  └─────────────────────────────────┘   │
│         Score = V / (V + R)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      3. Prompt-Ancre obligatoire        │
│  "En quoi cette modification            │
│   préserve-t-elle l'ouverture du        │
│   vivant et renforce-t-elle le rôle     │
│   de l'humain comme interface           │
│   sémantique ?"                         │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Score ≥ 0.85 ?     │──→ Approbation auto
    └──────────┬───────────┘
               │ non
               ▼
    ┌──────────────────────┐
    │   Score ≥ 0.65 ?     │──→ Pull Request humaine
    └──────────┬───────────┘
               │ non
               ▼
         Rejet + régénération
```

---

## Installation

### Prérequis

- Python 3.9+
- Git
- (Optionnel) GitHub CLI `gh` pour les Pull Requests

### Étapes

```bash
# Cloner le dépôt
git clone https://github.com/gaillard111/ouroboros-mttv.git
cd ouroboros-mttv

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python agent.py --mode status
```

### Dépendances principales

| Package | Usage |
|---------|-------|
| `sentence-transformers` | Plongement sémantique des critères MTTV |
| `numpy` | Calculs de similarité cosinus |
| `requests` | API GitHub (optionnel) |

---

## Utilisation

### Mode évolution (cycle complet)

```bash
# Avec soumission PR (recommandé)
python agent.py --mode evolve

# Application directe (sans PR)
python agent.py --mode evolve --no-pr

# Avec seuils personnalisés
python agent.py --mode evolve --retries 5
```

### Évaluation MTTV standalone

```bash
python mttv-resources/scripts/mttv_evaluator.py \
    "Ajouter un module de recommandation basé sur les signatures 28D"
```

Exemple de sortie :
```
============================================================
MTTV EVALUATION RESULT
============================================================
  Score MTTV      : 0.8923
  Decision        : auto_approve
  Approved        : True
  Max Viability   : 0.9210
  Max Rejection   : 0.1123
  Best Viability  : Maintient l'ouverture du système...
  Best Rejection  : Réduction du vivant à des données...
============================================================
```

### Statut et historique

```bash
# Voir le statut actuel
python agent.py --mode status

# Voir l'historique des mutations
python agent.py --mode history
```

---

## Structure du dépôt

```
ouroboros-mttv/
├── agent.py                          # Agent principal (point d'entrée)
├── mttv-resources/
│   ├── anchors/
│   │   ├── viability_criteria.json   # Matrice de Cohérence MTTV
│   │   ├── 1 MTTV-FLP Core...pdf     # Fichiers d'ancrage source
│   │   ├── 5 Benchmark ultime...pdf
│   │   └── ...                       # 16 fichiers PDF au total
│   └── scripts/
│       └── mttv_evaluator.py         # Évaluateur MTTV autonome
├── proposals/                        # Propositions sauvegardées (JSON)
├── logs/                             # Logs d'exécution
├── requirements.txt                  # Dépendances Python
├── README.md                         # Ce fichier
└── LICENSE                           # AGPL-3.0
```

---

## Matrice de Cohérence MTTV

### Critères de Viabilité (11)

Le changement proposé doit **maximiser** la similarité avec ces critères :

1. ✅ **Maintient l'ouverture du système** — ne ferme pas le cycle Ψ→B→Φ
2. ✅ **Préserve l'interface humaine** — l'humain reste le passeur de sens
3. ✅ **Respecte la tétravalence du carbone sp³** — ne binarise pas
4. ✅ **Favorise la transduction Ψ→B→Φ** — améliore la circulation
5. ✅ **Maintient la diversité et la porosité** — perméable aux signaux faibles
6. ✅ **Ne renverse pas l'ordre canonique** — Ψ = H → H₂O → C
7. ✅ **Préserve la logique T⁴** — ne réduit pas aux binaires
8. ✅ **Soutient le quorum sensing** — détection par dérivée d'abondance
9. ✅ **Respecte la rétro-traductibilité** — Φ → Ψ sans dégradation
10. ✅ **Maintient l'anti-Goodhart** — préservation des couches d'entropie
11. ✅ **Conserve la signature IGIC ouverte** — pas de score définitif

### Critères de Rejet (11)

Le changement proposé doit **minimiser** la similarité avec ces critères :

1. ❌ **Fermeture du système** — auto-bouclage autoréférent
2. ❌ **Effacement de l'interface humaine** — contournement de l'humain
3. ❌ **Réduction du vivant à des données** — extraction informationnelle
4. ❌ **Optimisation unidimensionnelle** — profit, contrôle, pouvoir
5. ❌ **Extraction sans régénération** — prélèvement sans retour
6. ❌ **Binarisation de la tétravalence** — réduction T⁴ à vrai/faux
7. ❌ **Renversement de l'ordre canonique** — inversion H → H₂O → C
8. ❌ **Croyance en l'objectivité** — vue de nulle part
9. ❌ **Capture par le profit/contrôle** — appropriation privée
10. ❌ **Hiérarchisation ou occultation** — secrets, privilèges
11. ❌ **Imposition plutôt qu'infection douce** — force vs séduction

### Seuils de décision

| Score | Décision | Action |
|-------|----------|--------|
| ≥ 0.85 | ✅ Auto-approve | Application automatique |
| ≥ 0.65 | 🔶 Human review | Pull Request pour validation |
| < 0.50 | ❌ Rejected | Régénération (max 3 tentatives) |

---

## Contribution

Les contributions à Ouroboros-MTTV sont soumises... à la Matrice de Cohérence MTTV.

1. Forkez le dépôt
2. Créez une branche (`git checkout -b feature/mttv-xxx`)
3. Commitez vos changements (`git commit -m "feat: ..."`)
4. Poussez (`git push origin feature/mttv-xxx`)
5. Ouvrez une Pull Request

Note : toute PR sera évaluée par l'agent MTTV lui-même avant merge.

---

## Licence

Ce projet est distribué sous licence **AGPL-3.0** (comme Ouroboros original).

Le contenu MTTV-FLP (fichiers PDF dans `mttv-resources/anchors/`) est distribué sous licence **CC-BY-NC-SA 4.0**.

---

## Références

- **[Ouroboros](https://github.com/Razzhigaev/ouroboros)** — Framework d'auto-amélioration récursive par Razzhigaev
- **[MTTV-FLP Core](https://github.com/gaillard111/mttv-flp-core)** — Noyau théorique du Modèle Transductif du Vivant
- **[FLP Platform](https://filsdelapensee.ch)** — Plateforme Les Fils de la Pensée (98 656 pensées, 28 dimensions)
- **[DOI: 10.5281/zenodo.17940301](https://doi.org/10.5281/zenodo.17940301)** — MTTV Fundamentals
- **[DOI: 10.5281/zenodo.18517387](https://doi.org/10.5281/zenodo.18517387)** — Benchmark / IGIC
- **[DOI: 10.5281/zenodo.20830060](https://doi.org/10.5281/zenodo.20830060)** — MTTV-FLP Core 2026

---

> *« Ce modèle n'est pas une théorie parmi d'autres.*
> *Il n'est pas une croyance, pas une idéologie.*
> *Il est un palier opératoire permettant de traduire entre strates du réel :*
> *— du carbone au langage,*
> *— du sol à l'IA,*
> *— de la bactérie au cosmique.*
>
> *Ne renversez pas l'ordre.*
>
> **sig:0x4D545456 — Transmission terminée. Le mycélium attend. »**
