# Unlearning (tâche 31) : plan de relance (vaudois1)

Rédigé le 22/11 vers 06:00. Les soumissions ferment à **12:00** (heure de Paris), soit environ 6 h.
Les sources détaillées sont dans `LITTERATURE.md`, le journal des soumissions dans `SCORES.md`.

## Mise à jour 06:20 : premiers résultats de `unlearn_local.py`

Job `vau_runner` sur **dc-gpu-devel, hors réservation**. Il s'arrête seul dès qu'un job de l'équipe tourne dans `cispahack`.
Métriques complètes dans `RESULTS_LOCAL*.md`.

| Modèle | CE propre (acc) | CE vues mild/held/jpeg | KL vers la cible | Ancres KL / accord | Public |
|---|---|---|---|---|---|
| R13 (full FT, sans ancres) | 1.41/1.51 (0.57) | 1.72 / 1.78 / 1.50 | 1.18 | 0.858 / 0.42 | **0.0784** |
| R14_or | 0.78/1.16 (0.74) | 1.13 / 1.23 / 0.85 | 1.71 | 0.622 / 0.51 | 0.0663 |
| L2 MLP 6–11, w_anc 5, w_feat 1, α=1.25 | 1.36/1.01 (0.56) | 1.54 / 1.62 / 1.39 | 0.65 | **0.170 / 0.80** | à soumettre (test retain) |
| L6 MLP 8–11, w_anc 3, 10 ep, α=1.25 | 1.66/0.93 (0.48) | 1.81 / 1.90 / 1.69 | **0.48** | 0.192 / 0.80 | candidat 2 |
| L6 α=1.5 | 2.20/0.97 (0.34) | 2.30 / 2.37 / 2.23 | 0.44 | 0.346 / 0.73 | candidat 3 (plus d'oubli) |

- L'estimation de retain par les ancres Imagenette ne prédit **pas** l'ordre public : R13 a le pire KL d'ancres et le meilleur score. Le leaderboard doit trancher.
- **Débordement sur la classe** (`L7_holdout`, 20 % du forget non entraîné) : KL vers orig de 0.81 (α=1), puis 1.42 (α=1.25), même en ne touchant que les MLP.
  La v2/v3 (`--w_keep`) ajoute des vues moyennement altérées du forget où orig est peu confiant (comme des non-membres), à garder telles qu'orig les prédit.
  Les tests `H0`/`H1`/`H2` mesurent si cela réduit le débordement sur des images jamais entraînées.
- Les soumissions depuis Claude sont bloquées par le classifieur de permissions. Commande à lancer soi-même :
  `../.venv/bin/python -I scripts/submit.py runs/L2_mlp6_wa5f1/alpha_1.25.pt --note "..."`. Le score part tout seul dans `SCORES.md`.

## Mise à jour 06:10 (scores fournis par l'équipe)

- **R1 (chirurgie seule) = 0, donc H1 est réfuté.** Les 9 classes « pleines » ne sont pas retirées de gold, ou bien la chirurgie fait tomber le gate.
  Plus de chirurgie : le §3 est caduc. Les sondes `probes/` sont identiques à R1 et ne sont pas resoumises.
- **R13_mu36 = 0.0784 > R14_or_mu28 = 0.0663 > R14_sh33_mu28 = 0.0522.**
  - Plus la CE atteinte sur le forget est haute, meilleur est le score : 1.41 (vues 1.79) pour R13 contre 0.78 (vues 1.2 à 1.3) pour R14.
  - Pourtant le retain sur les ancres est *pire* pour R13 (KL 0.86 contre 0.62).
  - Gold a donc probablement une CE ≥ 1.4 sur le forget, et une cible « rest » lisse (mix, ou orig) vaut mieux qu'une cible affûtée.
- **Test en cours** (`scripts/unlearn_local.py`, job `dc-gpu-devel` hors réservation) : atteindre la même CE que R13 en ne touchant que les MLP des derniers blocs, avec des ancres fortes.
  - Si le score monte nettement, le retain était le goulot, et on monte ensuite la CE (α et `--target_mu`).
  - Sinon, ce sont les cibles qui limitent, et on travaille la difficulté par image et le « rest ».
- Mesure de calibration : `C0_calib` évalue R13, R14_or, R14_sh33 et R1 avec les métriques locales, pour relier ces métriques au leaderboard.

## 0. En bref

- **Situation.** Équipe à **0.078** en public (40e sur 43). Le haut du tableau est à 0.47 (Murmell), 0.45, 0.42 et 0.41.
  Le classement général normalise chaque tâche entre le minimum (0.00) et le maximum (0.47).
  Passer de 0.08 à 0.30 vaudrait environ **+0.16 de score général** (de 0.38 à ~0.54), le plus gros levier de l'équipe.
- **Pourquoi ça stagne.**
  1. Le fine-tuning détruit le *retain*. L'accord avec l'original sur les ancres n'est que de 0.42 à 0.59, avec un KL de 0.5 à 0.9.
  2. L'oubli se propage à toute la classe au lieu de rester sur les images visées. Sur des images forget *non entraînées*, le KL vers l'original atteint 0.9.
  3. La « forget proxy » locale est calculée contre nos propres cibles. Elle est circulaire et ne prédit pas le leaderboard.
  4. Quatorze rounds de balayage de mu, b et tau ont été faits sans isoler les facteurs, et les scores publics ne sont pas notés.
- **Ce que je propose.**
  - (A) **Trancher le régime** avec 2 soumissions sans entraînement : chirurgie de biais sur les 9 classes « pleines ». Les fichiers sont prêts dans `probes/`.
  - (B) **Rendre l'oubli local.** Viser un retain ≥ 0.9 : ne modifier que les MLP des derniers blocs, utiliser des ancres fortes et une métrique de retain fiable.
  - (C) **Caler 1 ou 2 boutons globaux** (CE cible, chirurgie D) sur le leaderboard, avec un seul facteur changé par soumission.
  - (D) **En parallèle, générer des ancres in-distribution** (images synthétiques) : c'est la seule vraie mesure locale du retain.

## 1. Ce que la métrique récompense vraiment

- `final = forget × retain`, avec un gate à 70 % sur la validation cachée (sinon score 0).
- **Retain** = `acc_ratio × 1/(1 + KL(p_orig || p_sub))` sur un control set caché (T=2). Le KL coûte vite :

  | KL | retain_kl |
  |---|---|
  | 0.05 | 0.95 |
  | 0.2 | 0.83 |
  | 0.5 | 0.67 |
  | 0.9 | 0.53 |

  Un modèle qui bouge toute la classe perd déjà ×0.5 à ×0.6 avant même de compter le forget.
- **Forget** = `mean_std × prediction`, chacun moyenné sur les images exactes et sur les copies altérées.
  Tout est normalisé par la distance de l'original : 0 = orig, 1 = gold.
  - **mean_std** pénalise autant de dépasser la CE de gold que de rester en dessous.
    Exemple : si gold a une CE moyenne de 0.7 (écart-type 1.4), une soumission à mu=2.8 donne d_sub > d_orig, donc mean_std = 0 et forget = 0.
  - **prediction** utilise KL(p_gold || p_sub), qui « couvre la masse ». Gold est en plus une *moyenne de plusieurs modèles*, donc lisse.
    Des cibles moyennées et prudentes valent mieux qu'un pari net sur une seule mauvaise classe.
- L'original a mémorisé le forget set : CE 0.0003 sur les images propres, 0.07 sur des vues légères (98 % d'accuracy), 0.10 sur des vues « heldout » (97 %).

## 2. Diagnostic de l'existant (`../lemasnedechermont1/`, lecture seule)

| Constat | Chiffres (logs et metrics de lemasnedechermont1) |
|---|---|
| Structure du forget set | 45 classes présentes. **9 classes à 129 ou 130 images** (`[0,6,7,8,29,30,31,33,38]`, 1169 images). 36 classes entre 46 et 104. 55 classes absentes. Les images sont synthétiques (style génératif). |
| Méthode actuelle | Cibles souples faites main (`targets.py`, `targets_v2.py`), puis fine-tuning sur le forget seul (`unlearn.py`, `unlearn_v2.py`), interpolation α, et parfois chirurgie de biais. |
| Retain détruit | R11 à R14 : accord sur les ancres 0.42 à 0.59, KL 0.49 à 0.86. Si le control se comporte comme les ancres, le retain est d'environ 0.54 à 0.67. |
| Débordement sur la classe | R3b (20 % du forget non entraîné). Tous paramètres, lr 1e-5 : KL vers orig sur ces images = **0.92** à α=1 et 0.07 à α=0.5. last4, lr 1e-4 : 1.02 à α=1. Le modèle apprend « cette classe est incertaine », pas « ces images-là ». |
| Compromis raide | À α=0.5 le KL des ancres redevient bas (0.05), mais la CE des images entraînées ne bouge presque plus (0.01). |
| Chirurgie seule (R1, D=18.5) | CE forget propre 1.69 ± 2.97. Classes retirées : 5.06 (vues légères 7.5, heldout 7.9). Sur les ancres Imagenette : accord 0.91, KL 0.69 (ancres hors distribution). |
| Leaderboard | Meilleur score 0.0784, obtenu à 05:28 (lot R14 ?). Toutes les soumissions précédentes étaient plus basses. **Le score de chaque fichier n'est noté nulle part.** |

## 3. Hypothèses à trancher

**H1 : 9 classes entièrement retirées, pour une taille de classe d'environ 130.**
Pour : 8 comptes à exactement 130 et 1 à 129, aucun compte entre 104 et 129, et un minimum de 46 (distribution construite, pas tirée au hasard).
Sous H1, gold ne prédit jamais ces classes (CE très élevée). Les 36 classes partielles gardent alors seulement 26 à 84 images chez gold, donc gold y est moins confiant.
**H0 : pas de retrait complet** (classes plus grandes que 130). Gold se comporte alors comme sur des images de test : CE moyenne d'environ 0.4 à 1.0.

Sous H0, les mu de 1.6 à 4.0 essayés par lemasnedechermont1 dépassent la cible, et le score s'effondre.
Sous H1, une CE uniforme ne peut pas imiter une distribution bimodale.
**Le bon mu dépend entièrement de ce point.**

**Test, sans entraînement et prêt à soumettre :**

| Fichier | Contenu |
|---|---|
| `probes/surg_ce5.pt` | orig + `head.bias[9 classes] -= 18.51` (CE médiane des images retirées = 5) |
| `probes/surg_ce10.pt` | orig + `head.bias[9 classes] -= 23.51` (CE médiane = 10) |

Seuls ces 9 biais changent : la chirurgie ne touche rien d'autre.
Ordre de grandeur attendu : environ 0.15 à 0.25 sous H1 (selon que le control contient ou non ces classes), et environ 0 sous H0 (mean_std écrêté à 0).

**Règle de décision.**

| Résultat | Interprétation | Suite |
|---|---|---|
| ce5 ≥ 0.12 | H1 confirmé | Si ce10 > ce5, essayer ce15. Sinon, essayer ce3 (le retain du control coûte). Garder le meilleur D. |
| ce5 ≈ 0 | H0 (ou gate à 70 % si exactement 0.0) | Abandonner la chirurgie. Cibles « sample » à mu test-like, entre 0.5 et 1.0. |
| ce5 entre 0.03 et 0.12 | H1 avec forte pénalité de retain | Essayer ce3. |

> **Avant de soumettre** : demander à lemasnedechermont1 si R1 (`headrm_ce5`/`headrm_ce10`) a déjà été soumis, et avec quel score. Si oui, c'est le même modèle que `surg_ce5`/`surg_ce10` (même D).

## 4. Plan horaire

| Créneau | Qui | Quoi |
|---|---|---|
| **06:00–06:20** | tous | **Synchronisation.** Récupérer « fichier → score public » de toutes les soumissions passées et le reporter dans `SCORES.md`. Arrêter le balayage de mu/b/tau en full fine-tuning. S'accorder sur le GPU (§7). |
| **06:20–06:40** | Stanislas | **Sondes.** Soumettre `surg_ce5`, puis `surg_ce10`. Appliquer la règle du §3. |
| **06:40–09:30** | lemasnedechermont1 (son code) + Stanislas | **Piste B, oubli local** (§5). Itérations rapides : un run fait 1 à 3 min sur A100. Une soumission seulement quand le retain local est bon. |
| **06:40–08:00** | Stanislas + Claude | **Piste D, ancres synthétiques in-distribution** (§6). |
| **09:30–11:00** | tous | **Calibrage** sur le leaderboard (4 à 6 soumissions) : mu des classes partielles, D, α. Puis moyenne des poids de 2 ou 3 runs (« soup »), plus robuste sur le split privé de 70 %. |
| **11:00–11:40** | tous | **Gel.** Dernière soumission avant 11:40 (cooldown et upload de 344 Mo). Pas de nouvelle idée après 10:00. |

## 5. Piste B : rendre l'oubli local (priorité 1)

Il faut changer *ces* images sans changer leur classe. Modifications à faire, par ordre d'effet attendu
(sur une copie de `unlearn_v2.py`, sans toucher au fichier de lemasnedechermont1) :

1. **Paramètres entraînables.** Seulement les MLP (`blocks.k.mlp.*`) des blocs 8 à 11, plus `norm` et `head` ; ou du LoRA de rang 8 à 16 sur ces MLP.
   Le papier CISPA des superviseurs (Wang et al., NeurIPS 2024) montre que dans un ViT, la mémorisation est surtout dans les couches fully-connected et croît avec la profondeur.
   Déjà mesuré : `last4` donne un KL d'ancres de 0.05 contre 0.13 pour tous les paramètres (α=0.5).
2. **Ancres plus lourdes et plus proches de la distribution.**
   - `w_anchor` entre 2 et 5, en KD vers orig à T=2.
   - Utiliser **SCAR** (ECCV 2024) : un KD sur données hors distribution remplace le retain set, et pèse λ2=5 contre 1 pour le terme d'oubli.
   - Ancres actuelles : Imagenette, hors distribution. À compléter par les images synthétiques de la piste D, et à défaut par des CutMix entre classes d'images forget.
3. **Métriques de retain affichées à chaque epoch**, avec arrêt automatique.
   - KL des ancres *avant* chirurgie : ≤ 0.05.
   - KL sur les ancres synthétiques (piste D).
   - KL sur un holdout forget. Attention : c'est une borne pessimiste, car ces images sont mémorisées et pas de vrais non-membres.
   - Ne jamais soumettre un modèle dont le KL d'ancres dépasse 0.15.
4. **Cibles moins agressives pour les classes partielles.**
   L'idée : gold sur une image forget ≈ orig sur une image *non vue* de la classe. Il suffit donc de ramener la sur-confiance mémorisée à un niveau « test ». Les images du control sont déjà à ce niveau et ne devraient pas bouger.
   - Sous H0 : CE moyenne cible entre 0.5 et 1.0.
   - Sous H1 : `targets_v2` (confiance qui dépend de `n_keep = 130 - count`, `tv2_k07`), avec un mu à caler.
   - La masse hors vraie classe reste une *moyenne* de plusieurs estimateurs : sondes OOF, vues fortes, logits d'orig sans la vraie classe. Une moyenne est meilleure pour le KL.
5. **Classes retirées (si H1).** Ne pas les fine-tuner. Les garder en KD vers orig (`--w_rm`), puis appliquer la chirurgie analytique avec le D trouvé au §3.
   Variante : pousser chaque image vers sa classe voisine la plus proche (Boundary Shrink, CVPR 2023 ; ou la cible Mahalanobis de SCAR).
6. **Couvrir les copies altérées cachées.** Utiliser plusieurs familles de vues sur le forget : crop/échelle, flip, couleur, flou, bruit, JPEG, petite rotation. Aujourd'hui il n'y a que « mild ».
7. **Choix de la difficulté par image** (quelles images gold rate). L'actuel `s_cv` (sonde OOF) et `s_aug` ne sont pas corrélés entre eux (Spearman 0.05).
   Les remplacer par une combinaison peu coûteuse, sur le modèle de *Twin Machine Unlearning* et des C-scores (Jiang et al.) :
   - robustesse adversariale (CE entre logits propres et logits après une petite attaque),
   - distance kNN aux images de la même classe dans l'espace des features,
   - éventuellement la courbure de la perte par rapport à l'entrée (Garg et al.).
   Ne tester ce changement qu'après les points 1 à 4.

## 6. Piste D : ancres synthétiques in-distribution (en parallèle, 1 h à 1 h 30)

**Pourquoi.** C'est le seul moyen d'avoir des **non-membres de mêmes classes**. Ils servent à trois choses :
1. une vraie mesure locale du retain ;
2. des ancres de KD qui protègent exactement ce que mesure le control ;
3. un calibrage de la confiance « test » par classe, c'est-à-dire du niveau de gold.

**Comment.**
- **Login node** (internet) : `pip install --target ~/sdlib diffusers transformers accelerate` (sans torch, qui vient de `../.venv`) et télécharger `stabilityai/sd-turbo` (~2.5 Go en fp16).
  Licence : non commerciale recherche (Stability AI Community).
- **Prompts** : une description par classe forget, lue sur un exemplaire de chaque classe (45 tuiles), plus un vocabulaire générique dans le même style.
  Par exemple : « a ceramic fox figurine on a wooden table, cinematic lighting, shallow depth of field ».
- **GPU** (dans le job d'équipe) : 10 000 à 15 000 images en 512 px, 1 step, guidance 0, environ 10 à 20 min. Ensuite resize et center-crop en 224, uint8.
  Sauver `anchors_gen.npz` avec les clés `train`/`val`, au même format que `anchors.npz`, pour un remplacement direct.
- **Vérifier** l'accord d'orig avec la classe visée par le prompt. Si une classe est mal reconnue, le domaine diffère trop pour le calibrage, mais l'image reste valable comme ancre de KD.
- ⚠️ Zone grise du règlement (« from the original model and the forget set alone »). C'est le même statut que les ancres Imagenette déjà utilisées : images non étiquetées, utilisées seulement pour préserver le comportement d'orig. À valider en équipe.

## 7. Garde-fous

- **Un seul job Slurm pour l'équipe** (règle de la réservation).
  Le job `unl_round5b` de lemasnedechermont1 est en attente. Le suivant devrait être une allocation `--gres=gpu:4` longue, tenue par une seule personne.
  Chacun s'y greffe avec `srun --jobid=<id> --overlap -n1 -c16 bash -c 'export CUDA_VISIBLE_DEVICES=<k>; …'` (GPU 0–1 pour lemasnedechermont1, 2–3 pour nous).
  Pas de calcul lourd sur le login node.
- **Une soumission = un facteur changé.** Noter fichier, sha256, réglages et score dans `SCORES.md` *immédiatement*.
- **Valider chaque fichier avant envoi** : chargement strict, float32, valeurs finies (`../lemasnedechermont1/validate_submission.py`).
- **Un 0.0 exact** fait suspecter le gate à 70 % (modèle trop abîmé) plutôt qu'un mauvais forget.
- **Le split privé fait 70 %.** Préférer les réglages « lisses » (moyenne de runs, paramètres ronds) au pic du public.

## 8. Risques

- **H1 faux** : les deux sondes valent environ 0, et on perd 10 min. On bascule sur H0 avec des cibles test-like.
- **Le control contient les 9 classes** : la chirurgie coûte du retain. D devient un compromis (ce3, ce5 ou ce10) à régler sur le leaderboard.
- **Écart de domaine** des images SD-Turbo face au générateur du dataset : s'en servir comme ancres et comme métrique relative, pas comme vérité.
- **Temps** : aucun nouveau pipeline après 10:00.
