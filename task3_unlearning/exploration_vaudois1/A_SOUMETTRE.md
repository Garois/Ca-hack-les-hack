# Candidats à soumettre (mis à jour 07:06)

Score à battre : **0.0868** (meilleur de l'équipe, lemasnedechermont1, 05:58, probablement R15_or_mu55).
Déjà essayé : L16 α=1.25 (oubli « local » uniforme) = 0.0554, donc **pas** cette famille.
Le script `scripts/submit.py` réessaie tout seul en cas de cooldown (~40 s entre deux soumissions de l'équipe).

## 1. Sonde décisive : les 9 classes « pleines » sont-elles retirées chez gold ?

R1 (chirurgie de biais) = 0 peut s'expliquer par le **gate à 70 %**, puisque supprimer 9 classes partout coûte environ 9 points de validation.
P9a n'oublie **que** les images forget de ces 9 classes (perte 3.25), garde les 36 autres classes inchangées, et protège les non-membres avec les vues à garder.

    ! ssh juelich 'cd "/p/scratch/training2666/denoncin1/Ca hack les hack/unlearning/vaudois1" && ../.venv/bin/python -I scripts/submit.py runs/P9a_local/alpha_1.5.pt --note "P9a sonde 9 classes"'

- **P9a ≈ 0** : hypothèse fausse ; rester sur l'oubli uniforme de lemasnedechermont1.
- **P9a nettement > 0** (≥ 0.05) : les 9 classes sont retirées chez gold. Soumettre ensuite une version plus forte :

| Fichier | Perte 9 classes (exactes / altérées) | Autres classes | Ancres KL | Dans la classe KL |
|---|---|---|---|---|
| `runs/P9d_mlp4/alpha_2.pt` | 4.67 / 4.36 | 0.04 | 0.107 | 0.278 |
| `runs/P9c_keep2/alpha_2.pt` | 4.94 / 4.76 | 0.05 | 0.114 | 0.362 |

  Puis combiner : oubli fort sur les 9 classes, plus l'oubli uniforme modéré de R13/R15 sur les 36 autres classes. Les cibles se font avec `scripts/make_targets_groups.py --muA 8 --muB 2`.

## 2. Anciens candidats « locaux » (moins prioritaires depuis L16 = 0.055)

`runs/L16_strongkeep5_ep10/alpha_1.5.pt`, `runs/L2_mlp6_wa5f1/alpha_1.25.pt`, `runs/L10_mlp6_ep8/alpha_1.5.pt`
