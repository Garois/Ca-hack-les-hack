# vaudois1 : dossier de Stanislas pour la tâche Unlearning

Ce dossier est séparé de `../lemasnedechermont1/`, qu'on lit sans jamais le modifier.

| Fichier | Contenu |
|---|---|
| `PLAN.md` | Diagnostic, hypothèses, plan horaire jusqu'à 12:00. Voir d'abord la section « Mise à jour » en tête |
| `LITTERATURE.md` | Solutions existantes et papiers (liens, code), avec ce qu'on en tire |
| `SCORES.md` / `scores.jsonl` | Journal des soumissions, rempli automatiquement par `scripts/submit.py` |
| `RESULTS_LOCAL.md` | Une ligne par run et par α : CE forget (propre, mild, heldout, jpeg jamais vu), KL vers la cible, ancres KL/accord, retain_est |
| `scripts/unlearn_local.py` | **Entraînement « oubli local »** : KD vers les cibles sur forget et vues, seulement certaines couches (`--train mlp8-11`…), masque SalUn optionnel, ancres KD (`--w_anc`), features (`--w_feat`), L2-SP. Option `--eval_ckpts` pour mesurer des .pt existants |
| `scripts/make_targets.py` | Cibles souples : CE moyenne `--mu`, étalement `--b`, proxy de difficulté (`cv`, `aug`, `mild`, `clean` ou une combinaison), mélange de la masse hors vraie classe (`--w_cv/--w_strong/--w_orig/--w_unif`), `--tau` |
| `scripts/runner.sbatch` | Job **dc-gpu-devel, hors réservation**. Exécute `queue/*.sh` sur 4 GPU et s'arrête seul dès qu'un job de l'équipe tourne dans la réservation (`queue/STOP` pour arrêter à la main) |
| `scripts/submit.py` | Valide puis soumet un .pt (clé lue dans `../submission.py`, jamais affichée) et journalise le score |
| `scripts/watch.py` | Affiche les nouveaux événements : jobs, erreurs, résultats |
| `scripts/make_surgery_probe.py`, `probes/` | Sondes de chirurgie de biais. **R1 = 0** : hypothèse H1 réfutée, ne pas resoumettre |

## Usage rapide (depuis ce dossier)

    # ajouter un run à la file du runner (s'il tourne)
    echo "python -I scripts/unlearn_local.py --tag L9 --targets artifacts/targets_u20_mu36.npz --train mlp8-11 --lr 1e-4 --w_anc 3" > queue/20_L9.sh
    # (re)lancer le runner si besoin (hors réservation)
    sbatch scripts/runner.sbatch
    # soumettre un candidat
    ../.venv/bin/python -I scripts/submit.py runs/L9/alpha_1.pt --note "L9 alpha1"
