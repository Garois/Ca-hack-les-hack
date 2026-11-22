# Plan d'amélioration après la v8 (22/11, 00:45)

## 1. Pourquoi la v9 (sonde StegaStamp) a été rejetée — étude chiffrée (`v9_study.py`, `headroom.py`)
- **Gain illusoire.** En validation croisée **imbriquée** (réglages choisis sur 4 plis, score sur le 5ᵉ), la moyenne
  sonde + U-Net fait **−0,0006** (IC 95 % : −0,009 à +0,008). Le +0,005 initial venait du biais de sélection
  (75 combinaisons testées contre 15).
- **La v8 est solide** avec le même protocole :
  - PixelSeal (v7) : **+0,086** (IC +0,068 à +0,104) ;
  - TrustMark (v8) : **+0,050** (IC +0,035 à +0,066).
- **Cause du rejet : la sonde StegaStamp est trop faible**, le U-Net la domine en AUC partout :

  | sous-type | AUC U-Net | AUC sonde |
  |---|---|---|
  | trous | 0,951 | 0,865 |
  | objets | 0,900 | 0,834 |
  | rectangles | 0,906 | **0,585** (presque aveugle) |

  La complémentarité (corrélation de 0,05 à 0,37 entre les cartes) ne compense pas cet écart : +0,038 sur les objets,
  −0,027 sur les rectangles, −0,005 sur les trous.

### Deux découvertes qui orientent la suite
- **La décision, pas le classement.** Le U-Net StegaStamp classe très bien les pixels (AUC 0,90 à 0,95), mais le mIoU
  plafonne à 0,63-0,74. Sur les rectangles, il prédit 2,6 % de l'image au lieu de 5 % (sous-segmentation). Avec le
  meilleur seuil choisi image par image (borne haute), la marge est de :

  | méthode | actuel | borne haute | marge |
  |---|---|---|---|
  | PixelSeal | 0,574 | 0,607 | +0,033 |
  | TrustMark | 0,577 | 0,612 | +0,035 |
  | StegaStamp | 0,690 | 0,740 | +0,050 |

  Les plus grosses marges sont sur les objets et les rectangles (+0,02 à +0,07).
- **Rectangles StegaStamp.** Le décodeur tolère qu'on déplace **un bord** (4 px → acc 0,84 ; 8 px → 0,75), mais pas
  une **translation** (4 px → 0,60). Il faut donc une précision d'environ 2 px sur le **centre** et seulement
  d'environ 8 px sur la **taille**. Or les boîtes actuelles du U-Net sont loin (erreur médiane de 72 px sur le pire
  bord). 55 % des rectangles StegaStamp sont décodables avec la boîte exacte.

## 2. Plan
**A. Couche de décision** (CPU uniquement, sur les prédictions hors pli déjà calculées, pour les 3 méthodes)
- **A1. Calibration.** Régression isotone de la probabilité vers la fraction réelle de pixels tatoués, par méthode
  et par régime. Ajustée sur 4 plis, appliquée au 5ᵉ.
- **A2. Décodage qui maximise le mIoU attendu.** Trier les cellules par probabilité calibrée q, puis choisir le
  nombre K de cellules à garder qui maximise
  ½·[ S_K / (K + T − S_K) + (N − K − T + S_K) / (N − S_K) ],
  avec S_K = somme des K plus grands q, T = Σq et N = 16 384. Pas de seuil fixe, **indépendant du régime**, donc
  il corrige aussi les erreurs de régime.
- **A3. Rectangularisation** (images partielles). Si la plus grande région prédite est rectangulaire (remplissage de
  sa boîte englobante ≥ τ), la remplacer par un rectangle élargi aux cellules où q ≥ q_bas, pour corriger la
  sous-segmentation.
- **A4. Nettoyage** : retrait des petites composantes, comme aujourd'hui.

**B. GPU (un seul job d'équipe, ~1 h 30)**
- **B1.** Probabilités test **par modèle** des U-Net (6 par méthode), pour pouvoir calibrer chaque modèle avant la
  moyenne.
- **B2.** Recherche de rectangle sur **toutes** les images partielles du train (PixelSeal, TrustMark), pour évaluer
  hors pli exactement comme en test.
- **B3 (conditionnel).** Recherche locale fine des rectangles StegaStamp : centre au pas de 2 px (±12 px) × taille au
  pas de 8 px (±16 px), puis affinage à 1 px. Points de départ = boîtes issues de A3.
- **B4 (optionnel).** TTA par translations de 2 px pour les sondes et les U-Net.

**C. Assemblage** de la v9 et estimation en validation imbriquée, puis mise à jour de `METHODE_V8.md`. Libération du
nœud.

**Règle d'adoption** de chaque brique : gain apparié en validation **imbriquée** > 0, **avec la borne basse de l'IC 95 %
> 0**, par méthode.

## 3. Challenge — faiblesses du plan et résolution de chacune
| # | faiblesse | résolution |
|---|---|---|
| W1 | Sur-ajustement de la couche de décision (170 images par méthode, plusieurs réglages) | Validation imbriquée pour **toute** sélection ; calibration non paramétrique ajustée hors pli ; ≤ 2 paramètres par brique ; règle d'adoption sur l'IC bootstrap |
| W2 | Les probabilités hors pli viennent d'**un** modèle, celles du test d'une **moyenne de 6** (moins tranchées) : une calibration apprise sur les premières transfère mal | Calibrer **chaque** modèle de test séparément avec la courbe de sa famille, **puis** moyenner (B1 fournit les sorties par modèle) ; contrôle : la fraction prédite par régime en test doit coller à celle du hors pli ; A2 est peu sensible à une légère sous-confiance |
| W3 | La recherche de rectangle remplace une partie des masques partiels en test, mais pas en hors pli : estimations faussées sur ce sous-type | B2 : lancer la recherche sur toutes les images partielles du train, puis évaluer avec exactement la même logique qu'en test |
| W4 | B3 dépend de boîtes de départ aujourd'hui très mauvaises (72 px) | **Condition d'entrée** : après A3, mesurer l'erreur des boîtes. On ne lance B3 que si ≥ 30 % des rectangles décodables ont leur centre à ±12 px. Seuil d'acceptation choisi pour 0 fausse détection sur le train. Limité à 45 min |
| W5 | Gains attendus modestes, risque de courir après le bruit | Ordre fixé par la marge mesurée (StegaStamp en premier) ; IC obligatoire ; on s'arrête dès qu'une brique ne passe pas |
| W6 | Règle « un seul job par équipe » et créneau GPU | Phase A entièrement sur CPU ; un seul job, demandé **seulement** quand A est fini ; libéré dès la fin de B |
| W7 | Usage du nœud de login | Analyses sur CPU courtes (< 10 min), 8-16 threads, un seul processus |
| W8 | Aucun retour du leaderboard : notre estimation peut diverger du test | Soumettre **v8** (référence publique) avant la v9 ; contrôles de cohérence sur le test (fractions par régime) |
| W9 | Les erreurs de régime (10 % des objets PixelSeal) restent coûteuses | A2 est indépendant du régime ; vérifier spécifiquement les images mal classées |
| W10 | La reproductibilité doit rester totale | Toute brique adoptée est décrite dans `METHODE_V8.md` (paramètres, contrôles) |

## 4. Journal
- 00:45 — étude terminée, plan rédigé. Démarrage de la phase A sur CPU. Le scratch refuse de nouveau l'écriture
  (quota) : le plan est dans `~/wm_out/PLAN_V9.md`.
- 00:47 — **A (CPU)** : décodage calibré « mIoU attendu », en validation imbriquée.
  - StegaStamp : **+0,021** (IC +0,013 à +0,030) ✅.
  - TrustMark : +0,005 (IC −0,001 à +0,010) ❌ seul.
  - PixelSeal : +0,003 (IC −0,002 à +0,009) ❌ seul.
  - Rectangularisation : aucun gain. Surface attendue : inférieure.
- 00:48 — **Condition d'entrée de B3 (W4)** : 25 % des rectangles StegaStamp décodables ont un départ à ±12 px
  (< 30 %) → **B3 non lancé**.
- 00:49 — Job GPU 15904096 (démarré immédiatement). **B1 + B4** : probabilités par modèle, sans et avec TTA
  (4 translations de 4 px), hors pli et test, pour les 4 familles (~1 min chacune). Contrôle : sans TTA = OOF
  stocké (≤ 0,0004).
- 00:51 — **TTA + décodage**, en validation imbriquée : StegaStamp **+0,026** (IC +0,016 à +0,036), TrustMark
  **+0,008** (IC +0,003 à +0,013), PixelSeal **+0,009** (IC +0,003 à +0,015) → **adopté pour les 3 méthodes**.
  B2 n'est pas nécessaire : les gains portent sur les trous et les objets, que les rectangles ne touchent pas.
- 00:53 — **v9 construite** : calibration par modèle, puis moyenne, décodage et rectangles.
  - Contrôle W2 : la baisse de la fraction prédite en test pour TrustMark et StegaStamp (presque tatoué) existe déjà
    en v8, et le décodage par modèle donne la même → propriété des images test, pas un artefact.
  - Nœud libéré à 00:54. `METHODE_V8.md` mis à jour (§5.9 et suivants).
