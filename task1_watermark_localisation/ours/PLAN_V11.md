# Plan v11 (22/11, 02:30) — nouveaux facteurs, plan repensé 3 fois avant d'implémenter

## Étude des nouveaux facteurs (`study_new.py`, `study_new2.py`, `study_texture.py`, `sam_gate.py`)
| facteur | résultat | conséquence |
|---|---|---|
| Remover officiel TrustMark (résidu de suppression, énergie et corrélation au motif attendu) | AUC 0,48-0,53 | sans signal, écarté |
| WMForger (modèle de préférence générique de Meta, carte par position à 512 et 768) | AUC 0,44-0,53 (StegaStamp objets : 0,27, soit 0,73 inversé) | sans intérêt, écarté |
| Décalage train/test | PixelSeal identique ; TrustMark et StegaStamp : tatouage global un peu plus faible en test | trous un peu plus grands en test, pas de décalage technique |
| Anatomie des trous (v9) | PixelSeal et TrustMark : 29 % manqués, précision 0,42-0,44 (StegaStamp 0,73) | c'est l'emplacement qui est faux, pas la quantité |
| Texture contre détection des trous | corrélation −0,07 à −0,14 ; la **taille** domine (+0,34 à +0,48) | ce n'est pas un manque physique de signal : la résolution des indices liés au message est trop faible (8×8) |
| Composantes multiples | rectangles : toujours 1 ; objets : 12-24 % en ont ≥ 2 | la recherche à un seul rectangle suffit |
| Visibilité des contours | objets 1,7-2,3 ; rectangles 0,9-1,0 ; trous 1,0 | un segmenteur ne peut aider que les **objets** |
| **SAM ViT-B** (segments « everything ») sur 14 objets | oracle **0,93-0,94** contre 0,59-0,75 actuellement ; vote simple +0,06 à +0,11, mais instable | les segments contiennent les objets ; **c'est leur sélection** qui compte |

## Plan v1
1. SAM pour les objets ;
2. sonde avec contexte + texture pour les trous ;
3. seuil des rectangles StegaStamp fixé sur données indépendantes.

**Réflexion 1** : l'hypothèse « trou manqué = zone plate » est fausse (mesure ci-dessus). Le point 2 doit viser la
**résolution des indices qui portent le message**, pas la texture.

## Plan v2
1. SAM pour les objets ;
2. **décodeur local du message** : une tête prédit les bits par cellule de 8 px à partir des couches fines du
   décodeur gelé ; son accord avec le message connu donne une carte 8× plus fine que la carte actuelle, entraînée
   conjointement avec la segmentation ;
3. seuil StegaStamp en priorité basse.

**Réflexion 2** : il y a peu d'information dans une cellule, mais la moyenne sur les bits et l'agrégation spatiale
de la tête dilatée compensent. Perte conjointe = segmentation + BCE des bits (cellules tatouées) + accord ramené
vers 0 (cellules propres). On juge en validation imbriquée.

## Plan v3 (final)
**Réflexion 3 (faisabilité, test SAM)** :
- SAM est installable sans toucher au venv d'équipe (`~/wm_out/pylib`, poids ViT-B md5 01ec64…).
- Le test confirme le potentiel (oracle 0,93).
- Les calculs lourds passent **par Slurm** : un lancement CPU sur le nœud de login n'a pas été autorisé.

**A. SAM sur les images classées partielles** (train hors pli et test, 3 méthodes)
- Masques « everything » (32 points par côté), ramenés à la grille 128.
- Variantes de sélection :
  - V1 : vote mean(q) ≥ τ, τ choisi en validation imbriquée ;
  - V2 : ajout glouton de segments tant que le mIoU espéré augmente (sans paramètre) ;
  - V3 : carte moyennée par plus petit segment, puis décodage du mIoU attendu (sans paramètre) ;
  - V4 : mélange α·q + (1−α)·q_segment, α choisi en validation imbriquée.
- Choix par méthode en validation imbriquée, avec contrôle par sous-type (rectangles StegaStamp).

**B. Décodeur local du message** (PixelSeal, TrustMark) : 5 plis + tout le train, 8 epochs, tête dilatée, TTA.
Évalué comme nouvelle famille, seule et en moyenne avec « v9 + dilatée ».

**C. Assemblage v11** et contrôle **combiné** en validation imbriquée, puis documentation et libération du nœud.

**Règle d'adoption** : borne basse de l'IC 95 % > 0, par méthode (inchangée).

| faiblesse | résolution |
|---|---|
| Sélection des segments qui sur-ajuste | variantes V2 et V3 sans paramètre ; V1 et V4 choisis en validation imbriquée |
| SAM nuisible sur les rectangles (contours invisibles) | contrôle par sous-type ; les rectangles trouvés gardent la priorité |
| Temps GPU | SAM ~20 min réparti sur 3 GPU ; décodeur local ~30 min ; un seul job, libéré à la fin |
| Calcul hors Slurm | tout dans le job (`srun --overlap`) |
| La v10 doit rester un repli | la v11 n'est construite que si le contrôle combiné est positif |

## Journal d'exécution
- 02:20 — Le test SAM lancé sur CPU sur le nœud de login n'a pas été **autorisé**. Il a été
  refait sur GPU dans un job court (15904493), libéré aussitôt. Oracle 0,93-0,94.
- 02:28 — Job 15904539 en file (démarré à 02:42). Lanceur automatique : segments SAM (3 GPU) et décodeurs locaux du
  message (1 GPU). Tout est fini à 03:15.
- 03:17 — **Étape 1** (cartes) : le décodeur local du message est rejeté (seul −0,012 / −0,004 ; avec « v9 + dilatée »
  +0,0005 / +0,003, IC contenant 0).
- 03:20 — **Étape 2** (SAM, partiel), sans rectangles :
  - PixelSeal : glouton +0,013 (IC > 0) ;
  - StegaStamp : mélange 0,5, +0,004 (IC > 0) ;
  - TrustMark : +0,0004 ❌. Correction W3 : SAM est pénalisé sur les rectangles TrustMark, alors qu'ils sont remplacés
    en test.
- 03:30 — Recherche de rectangles sur **toutes** les images partielles du train (TrustMark : 40 acceptés ; PixelSeal :
  16), puis réévaluation **avec remplacement par les rectangles**, comme en test :
  - PixelSeal : **+0,013** (IC +0,004 à +0,022, glouton dans les 5 plis) ✅ ;
  - TrustMark : +0,006 (IC −0,0015 à +0,014) ❌.
- 03:33 — **v11 construite** : v10 + SAM pour PixelSeal (glouton) et StegaStamp (mélange 0,5). TrustMark est identique à
  la v10 (0,000 %). Cellules changées : PixelSeal 1,3 %, StegaStamp 0,5 %. Nœud libéré à 03:34 (52 min).
  `METHODE_V8.md` mis à jour (§5.11, mise en place de SAM, règle Slurm).
