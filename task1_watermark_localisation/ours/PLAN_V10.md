# Plan v10 (22/11, 01:05) — même processus : étude → plan → challenge → exécution avec validation imbriquée

## 1. Étude post-v9 (`study_v10.py`, hors pli avec TTA, décision v9)
| sous-type | PixelSeal v9 / borne seuil | TrustMark | StegaStamp | part de la perte (PS / TM / SS) |
|---|---|---|---|---|
| trous | 0,603 / 0,629 | 0,602 / 0,628 | 0,753 / 0,786 | **0,194 / 0,195** / 0,122 |
| objets | 0,644 / 0,689 | 0,614 / 0,640 | 0,670 / 0,712 | 0,088 / 0,093 / 0,080 |
| rectangles (hors pli, sans rectangles trouvés) | 0,479 | 0,520 | 0,684 | 0,132 / 0,124 / 0,082 |

Ce que l'étude montre :
- **La décision est presque au plafond** (borne haute du seuil à +0,02-0,04). Les gains doivent venir des
  **cartes**, donc des modèles.
- **Trous PixelSeal et TrustMark**, selon leur taille :
  - < 2 % : ≈ 0,50 (indétectables) ;
  - 2-5 % : 0,57-0,58 ;
  - 5-20 % : 0,64-0,70 ;
  - > 20 % : 0,62-0,64, en baisse.

  Le champ récepteur de la tête de sonde (4 conv3×3 sur la grille 64, soit ~70 px) est trop petit pour un grand
  trou : son centre n'est entouré que de « trou ».
- **Points de départ des rectangles StegaStamp** avec les cartes v9 : centre à ±24 px dans 43 % des cas, taille
  relative à ±0,6 dans 55 %, les deux dans 32 %. 55 % des rectangles sont décodables avec la boîte exacte.

## 2. Plan
**A (CPU). Filtre guidé par l'image**
- Lisser la carte calibrée q en suivant les bords de l'image (guide = image ramenée à 128, rayon r ∈ {2, 4, 8},
  ε ∈ {1e-3, 1e-2}), avant le décodage.
- Option « off » par régime.
- Cible : les objets, dont les contours sont visibles (rapport de gradient de 1,7 à 2,3) ; les trous ont des
  contours invisibles.

**B (GPU, un seul job, ≤ 1 h 30)**
- **B1. Sonde v2 (PixelSeal, TrustMark)** : tête **dilatée** (dilatations 1, 2, 4, 8 → champ récepteur ~250 px) et
  8 epochs fixes. 5 plis + tout le train, puis TTA.
- **B2. Graines** : 2 graines de plus pour la sonde actuelle (5 plis + tout le train chacune). On évalue la moyenne
  des graines.
- **B3. StegaStamp** : une 2ᵉ graine de U-Net (5 plis + tout le train), avec TTA. On évalue la moyenne des 2 graines.
- **B4. Rectangles StegaStamp**, recherche guidée :
  - centre sur une grille de 4 px à ±24 px autour du départ v9 (erreur ≤ 2 px, la tolérance de translation) ;
  - hauteur et largeur = départ × {0,6 ; 0,75 ; 0,9 ; 1,1 ; 1,3 ; 1,6} ;
  - soit ~6 000 recadrages par image ;
  - puis affinage au pixel des 3 meilleurs (chemin exact) ; acceptation si acc ≥ 0,8.

**C. Assemblage** de la v10 avec les seules briques adoptées. Contrôle **combiné** en validation imbriquée, puis
documentation, puis libération du nœud.

**Règle d'adoption** (inchangée) : gain apparié en validation imbriquée, borne basse de l'IC 95 % > 0, par méthode.

## 3. Challenge — faiblesses et résolutions
| # | faiblesse | résolution |
|---|---|---|
| W1 | Beaucoup de candidats, donc risque de fausse découverte | Chaque brique est évaluée séparément (IC > 0), **puis** le pipeline combiné est revérifié en validation imbriquée ; à égalité, on garde le plus simple |
| W2 | Comparer une tête v2 à la v1 est faussé par le hasard de la graine | B2 mesure la variance entre graines. La v2 doit battre la **moyenne des graines** v1 à coût égal ; sinon on combine (v1 + v2) |
| W3 | Budget GPU et règle d'équipe | Un seul job, 4 GPU en parallèle (GPU0 : PS v2 puis PS graines ; GPU1 : TM v2 puis TM graines ; GPU2 : graine U-Net SS ; GPU3 : B4 puis graines restantes). Libéré à la fin |
| W4 | Les graines partagent le même synthétique, donc peu de diversité | Accepté (gain attendu modeste) ; mesuré en validation imbriquée |
| W5 | B4 : fausses détections en prenant le maximum sur 6 000 candidats | acc ≥ 0,8 sur 100 bits : queue binomiale ~5,6e-10 par candidat → ~3e-6 par image. Vérifié **0 fausse détection** sur les objets partiels du train ; acceptation seulement en régime partiel |
| W6 | B4 modifie des masques partiels de StegaStamp : il faut une évaluation exacte | Recherche lancée sur **toutes** les images partielles du train (départs hors pli), évaluée avec la même logique qu'en test |
| W7 | Filtre guidé : nuisible sur les trous (contours invisibles) | Choix par régime avec option « off » en validation imbriquée |
| W8 | Une nouvelle carte demande une nouvelle calibration | La calibration isotone est réajustée dans chaque pli de l'évaluation imbriquée (déjà en place) |
| W9 | La v9 doit rester un repli sûr | La v10 n'est construite que si le contrôle combiné est positif ; la v9 reste en place |
| W10 | Reproductibilité | `METHODE_V8.md` mis à jour avec chaque brique adoptée |

## 4. Journal
- 01:00 — **A (CPU), filtre guidé** sur les cartes v9, en validation imbriquée :
  - PixelSeal +0,007 (IC +0,004 à +0,011) ✅ ;
  - TrustMark +0,008 (IC +0,004 à +0,012) ✅ ;
  - StegaStamp 0 (« off » choisi partout).
- 01:03 — Job 15904163 (démarré immédiatement). B1 à B4 lancés en parallèle sur les 4 GPU ; fin à 01:47.
- 01:50 — **Évaluation imbriquée de B**, contre la v9 :

  | méthode | variante | gain | IC 95 % |
  |---|---|---|---|
  | PixelSeal | « v9 + dilatée + filtre » | **+0,012** | +0,007 à +0,018 |
  | TrustMark | idem | **+0,0175** | +0,012 à +0,024 |
  | StegaStamp | 2ᵉ graine U-Net | +0,002 | −0,008 à +0,012 ❌ |
  | StegaStamp | rectangles guidés | +0,0085 | −0,001 à +0,018 ❌ |

  Pour les rectangles StegaStamp : 19/44 vrais rectangles à acc ≥ 0,98, mais 1 fausse détection à 0,81 sur un objet.
- 01:53 — **W1** :
  - la variante choisie *dans* la validation imbriquée est « v9 + dilatée + filtre » dans les 5 plis, pour les deux
    méthodes → aucun biais de sélection ;
  - seuil des rectangles StegaStamp choisi en validation imbriquée : +0,0084 (IC −0,0009 à +0,018) → toujours
    rejeté. La condition W5 (0 fausse détection) n'est pas remplie non plus.
- 01:56 — **v10 construite** : StegaStamp identique à la v9 (0,000 %) ; PixelSeal 2,3 % et TrustMark 2,5 % de cellules
  changées. Les fractions en test sont cohérentes. Nœud libéré à 01:57 (54 min). `METHODE_V8.md` mis à jour (§5.10).
