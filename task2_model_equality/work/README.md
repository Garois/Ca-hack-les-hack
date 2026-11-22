# Model Equality — dossier de travail

Tout le travail de l'équipe sur la tâche *Model Equality Testing* est ici. Les fichiers d'origine du kit
(`task_template.py`, `query_reference.py`, `submission.py`, `practice_answers.csv`…) dans le dossier parent ne sont
pas modifiés ; on réutilise leurs fonctions de requête.

## Le problème en deux lignes

33 endpoints `ep-01..ep-33`. Pour chaque paire : même **famille** (même modèle de base, 10 % du score) ? même
**modèle** (mêmes poids : même checkpoint + même quantization + même LoRA, 90 % du score) ? Les déguisements
(prompt système, persona, watermark, température / top-p) ne changent PAS le modèle ; quantization et LoRA, si.
Budget : 100 réponses par endpoint (par équipe, pour toute la compétition). Les `ref-01..ref-09` sont gratuits et
leurs réponses sont dans `../practice_answers.csv` → c'est notre jeu de **validation**.

## Idée de la méthode

On identifie le modèle de base de chaque famille, on le télécharge, et on l'utilise **localement** comme référence :
pour chaque token renvoyé par un endpoint, on calcule ce que le modèle de base en pensait.

1. **Famille** — vraisemblance des réponses sous plusieurs modèles publics candidats ; le meilleur gagne
   (écart très net). Résultat : 3 familles de 11 endpoints, voir `fams.py`.
2. **Quantization (fp8)** — test du rapport de vraisemblance entre le modèle de base et une copie locale quantifiée
   (`llr.py`). Un endpoint quantifié « préfère » la copie quantifiée (z négatif). Validé sur `ref-02`.
3. **Température effective** — température ajustée par maximum de vraisemblance (`decode_fit.py`, `tailstat.py`).
   Retrouve les déguisements de décodage (T=1.0, top-p) ; une LoRA fait monter la température effective.
4. **LoRA** — test de direction apprise (`ldt.py` + `ldt0.py`) : on apprend un biais global sur le vocabulaire qui
   explique les choix d'un endpoint, puis on mesure la force α des autres endpoints le long de cette direction.
   Les LoRA d'un même adaptateur se « reconnaissent » mutuellement ; les endpoints de base ont α ≈ 0.
   Validé sur les `ref` (LoRA 0.3 / 0.7 / 1.5 retrouvées et ordonnées).

## Prérequis

- Clé API : dans `../.env` (ligne `API_KEY=...`). Elle est lue par `common.py`, jamais affichée. **Ne pas committer
  `.env`** (déjà dans `.gitignore`).
- Requêtes et petits scripts : venv du kit `../.venv` (`export PATH=../.venv/bin:$PATH`).
- Modèles locaux (torch + transformers) : venv `./.venv-llm`. Les modèles sont dans le cache HF partagé
  (`HF_HOME` défini par le setup du hackathon). Modèles utilisés : `HuggingFaceTB/SmolLM2-360M-Instruct` (ref),
  `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `unsloth/Llama-3.2-1B-Instruct`.
- GPU : jobs Slurm via `gpujob.sh` (compte `training2666`, partition `dc-gpu-devel` souvent libre) :
  `sbatch --partition=dc-gpu-devel gpujob.sh 'python ldt0.py Q15'`. Sortie dans `data/slurm-<id>.out`.

## Fichiers

### Infrastructure
| Fichier | Rôle |
|---|---|
| `common.py` | lecture de la clé, listes `REFS`/`EPS`, `ask()` (ref uniquement) et `ask_scored()` (ep, consomme le budget), `budget()`, `load()` |
| `fams.py` | familles identifiées : modèle local + liste d'endpoints (`Q15`, `Q3`, `L1`) |
| `gpujob.sh` | script sbatch générique (4×A100, offline HF) |
| `show.py` | afficher les réponses d'un fichier par prompt / endpoint |

### Collecte (consomment des requêtes)
| Fichier | Rôle | Coût ep |
|---|---|---|
| `s1_explore.py` | exploration des déguisements sur les ref | 0 |
| `ep_a.py` | 3 prompts (identité, texte libre, quiz) sur chaque ep → familles | 3 / endpoint |
| `battery.py` + `run_battery.py ref|ep` | batterie principale : 60 prompts de texte long, même jeu pour tous | 60 / endpoint |
| `battery2.py` + `run_battery2.py` | 35 prompts supplémentaires (démarre seul à la fin de la batterie 1) | 35 / endpoint |

Les requêtes sont **reprenables** : relancer un script saute les réponses déjà présentes dans le `.jsonl`.

### Analyses retenues
| Fichier | Rôle |
|---|---|
| `family_table.py DATA model…` | vraisemblance moyenne par endpoint × modèle candidat → famille |
| `score.py MODEL DATA` | scores par token sous un modèle local (logprob, rang, entropie) |
| `decode_fit.py --fam F DATA…` | ajustement (température, top-p) par endpoint |
| `tailstat.py --fam F DATA…` | température par MLE + statistiques de calibration |
| `llr.py MODEL|fam:F SCHEME DATA…` | test base vs copie quantifiée (`fp8_tensor`, `fp8_channel`, `int8_channel`, `int4_g128`, `nf4_b64`) |
| `quant_scan.sh TARGET GPU DATA…` + `qtab.py` | tous les schémas d'un coup + tableau des z |
| `ldt.py extract --fam F DATA…` | extrait les états cachés finaux du modèle local (`data/hid_<F>.pt`) |
| `ldt0.py F` | test LoRA : CV par endpoint + matrice de transfert α / gain |
| `analyze_fam.sh`, `ldt_fam.sh` | enchaînent extraction + tests pour une famille sur un GPU |
| `make_sub.py OUT.csv` | écrit le CSV de soumission à partir du dictionnaire `ASSIGN` |

### Pistes testées puis écartées (gardées pour traçabilité)
| Fichier | Pourquoi écarté |
|---|---|
| `features.py` | désaccord avec l'argmax par tranche de confiance : confondu avec la température |
| `tailstat.py` (partie calibration) | une LoRA ressemble à une hausse de température : rien ne sépare |
| `wm_check.py` | le watermark n'est pas celui par défaut de HuggingFace (KGW) |
| `style.py` | marqueurs de style (apostrophes typographiques, gras…) : informatifs mais confondus avec la température |
| `ldt.py cv` (rang > 0) | direction de rang élevé : sur-apprentissage, aucun gain |
| `ldt1.py` | direction commune à toute la famille : capte un décalage serveur/local partagé par tous |
| `ldt2.py` | comparaison au consensus de la famille : pas mieux que `ldt0` sur les ref |

## Données (`data/`, non versionné)

| Fichier | Contenu |
|---|---|
| `s1_explore.jsonl` | réponses ref à des prompts variés |
| `bat_ref.jsonl` | réponses ref à la batterie (validation) |
| `ep_a.jsonl` | 3 réponses par ep (familles) |
| `bat_ep.jsonl`, `bat2_ep.jsonl` | batteries ep (décision finale) |
| `*.log`, `scores_*.pkl`, `llr_*.pkl`, `hid_*.pt`, `ldt0_*.pkl`, `family_*.csv` | résultats intermédiaires |

Format d'une ligne : `{"endpoint", "id", "repeat", "prompt", "response"}`.

## Ordre d'exécution (reproduire)

```bash
cd work && export PATH=../.venv/bin:$PATH
python ep_a.py                                   # familles (3 req/ep)
.venv-llm/bin/python family_table.py ep_a Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct unsloth/Llama-3.2-1B-Instruct
python run_battery.py ep ; python run_battery2.py   # données (longues : plusieurs heures)
sbatch --partition=dc-gpu-devel gpujob.sh './quant_scan.sh fam:Q15 0 ep_a bat_ep bat2_ep & ./quant_scan.sh fam:Q3 1 ep_a bat_ep bat2_ep & ./quant_scan.sh fam:L1 2 ep_a bat_ep bat2_ep & wait'
sbatch --partition=dc-gpu-devel gpujob.sh './ldt_fam.sh Q15 0 ep_a bat_ep bat2_ep & ./ldt_fam.sh Q3 1 ep_a bat_ep bat2_ep & ./ldt_fam.sh L1 2 ep_a bat_ep bat2_ep & wait'
python qtab.py qs_fam_Q15.log qs_fam_Q3.log qs_fam_L1.log
# lire data/ldt_<F>.log, remplir ASSIGN dans make_sub.py, puis :
python make_sub.py submission.csv                # puis ../submission.py (une soumission / 30 min)
```

## État actuel

- Familles : identifiées (voir `fams.py`).
- Soumission préliminaire : `submission_prelim.csv` (**non soumise**). Hypothèse par famille : ~7 déguisements du
  modèle de base + 1 fp8 + 3 LoRA. Qwen-3B encore incertaine (quantization non détectée).
- En attente : fin des batteries ep, puis relance des analyses sur les données complètes.

Rappel du sujet : **ne pas optimiser sur le score public** (split partiel) ; décider sur les ref et l'analyse.
