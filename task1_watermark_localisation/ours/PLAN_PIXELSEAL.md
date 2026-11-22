# Plan PixelSeal — de ~0,55 à ≥ 0,65 (21/11, 23:15)

## Diagnostic de départ
Bilan des pertes PixelSeal (train, pipeline v6) :

| sous-type | % images | score | perte |
|---|---|---|---|
| presque tatoué, trou en forme d'objet | 49 % | 0,476 | **0,256** |
| partiel, rectangle (tatoué à son échelle) | 25 % | 0,477 → 0,658 avec rect | 0,132 |
| partiel, objet transplanté (pleine échelle) | 25 % | 0,527 (10 % mal classées en « presque tatoué ») | 0,117 |

Les trous et les objets transplantés (~75 % des images, 0,37 de perte) ont un point commun : un tatouage
**pleine échelle** présent ou absent localement. La seule carte actuelle (logits par position du détecteur) est en
**8×8 natif**, trop floue pour des trous de ~10 % de l'image.

Faits établis sur PixelSeal (lecture du code videoseal) :
- résidu = 0,2 × JND(x) × agrandi×2( UNet(luminance de x à 256, message) ) ;
- donc **gris (R = G = B)**, lisse à l'échelle de 2 px, d'amplitude ~1 niveau de gris ;
- détecteur = ConvNeXt-tiny à 256, sortie 8×8 (`pixelwise: False`), canal 0 = détection, canaux 1 à 256 = bits.

Le test de re-tatouage du soir était biaisé : résidu arrondi à l'entier, passe-haut de 3 px (qui tue un motif lisse
à 2 px) et RGB au lieu de la luminance.

## Hypothèse centrale
Une carte d'indices « message présent » **au pixel près** localise les zones tatouées pleine échelle (trous et objets).

Signaux candidats :
- **S1 — gradient × entrée** : gradient de Σ s_k·logit_k du vrai détecteur, multiplié par la luminance passe-bande
  à l'échelle 256. L'alignement géométrique est assuré par le détecteur lui-même, qui décode les vraies images à 99 %.
- **S2 — filtre adapté par re-tatouage**, en flottant : corrélation locale entre la luminance passe-bande de x et le
  résidu r' = embed(x, m) − x, sans arrondi.
- **S3 — canal 0 de détection par position**, indépendant du message.
- **S4 — référence** : la carte par position actuelle (32×32 par shift-and-stitch).

## Étapes
- **P0 (immédiat)** : vérifier les jobs de l'équipe, puis soumettre **un seul** job de 2 h (4 GPU) **depuis le dossier
  de l'équipe**. Toutes les sorties vont dans `~/wm_out`, car le quota du scratch est instable.
- **P1 (diagnostic, ≤ 30 min, CPU pendant l'attente du GPU)** : `exp_ps_maps.py` calcule S1 à S4 sur les vraies images
  PixelSeal du train et sur ~100 images synthétiques (contrôle). Pour chaque sous-type, il mesure l'AUC par image,
  le mIoU avec un seuil relatif global (réaliste) et le mIoU avec le meilleur seuil par image (borne haute).
  **Porte de décision** : le seuil global doit battre la règle actuelle d'au moins +0,05 sur les trous ou sur les
  objets. Sinon → plan de repli F.
- **P2 (construction, ≤ 45 min GPU)** : calculer la ou les cartes retenues pour le train, le test et synth2
  PixelSeal (fichier `feats3`). Comparer en hors pli :
  - (a) une règle indépendante du régime : masque = carte > τ × max robuste, plus morphologie, ≤ 3 paramètres ;
  - (b) un U-Net à 5 plis avec les nouveaux canaux (`n_aux` déduit du checkpoint).
  
  Choisir par régime avec `oof.py` ; redécider le régime à partir de la carte.
- **P3 (≤ 15 min)** : v7 = v6 avec PixelSeal remplacé, en gardant les rectangles. Estimation hors pli, comparaison
  avec v6, **libération du nœud**.
- **F (repli)** : (i) rectangles PixelSeal : recherche fine autour des indices de la carte et du U-Net ; (ii) seuil
  calibré par image pour la règle sur carte. Gain attendu faible. Limité à 30 min, sinon on s'arrête à v6.

## Challenge n° 1 — le signal est-il réel ?
- Le re-tatouage (S2) a déjà échoué une fois. Si ce n'est pas un biais de mesure, la cause probable est un
  **désalignement géométrique** : les vraies images auraient été tatouées à une autre taille, puis recadrées ou
  redimensionnées. Le détecteur y est robuste, pas un filtre au pixel. → **Ajout** : contrôle sur synthétique
  (aligné par construction), pour séparer « méthode cassée » et « données désalignées ».
- Le gradient × entrée (S1) est connu pour être bruité, et il surligne les textures quel que soit le tatouage.
  → **Ajout** : normalisation par un **message nul**. Même calcul avec K = 4 messages aléatoires, puis carte
  z = (vrai − moyenne nulle) / écart-type nul. Cela retire le biais lié au contenu. Le même contrôle s'applique à S2.
- Les logits du détecteur saturent-ils sur les images très confiantes ? → Utiliser Σ s·logit, linéaire, sans
  sigmoïde.
- Si S2 réussit sur le synthétique mais échoue sur le réel, faire une recherche d'alignement (échelle et
  recadrage) **seulement si S1 échoue aussi**, limitée à 20 min.

## Challenge n° 2 — est-ce faisable dans le temps et avec les ressources ?
- 17 jobs attendent dans la réservation, donc l'attente du GPU est inconnue (peut-être plus d'une heure).
  → **P1 tourne sur CPU** pendant l'attente, sur un sous-ensemble (~60 images). Le détecteur à 256 sur CPU prend
  ~0,3 s par passe. La décision est prise **avant** d'avoir le GPU.
- Quota : cartes 128×128 en float16 × ~8 500 images ≈ 250 Mo par canal → `~/wm_out` (20 Go disponibles).
- Règle d'équipe : un seul job, soumis depuis le dossier de l'équipe, durée de 2 h, libéré dès que possible.
  **Arrêt dur à 02:00**, pour garder de la marge avant l'échéance de 14:00.
- Compatibilité : les anciens U-Net ont 3 canaux auxiliaires. → `n_aux` est déduit des formes du checkpoint, et un
  fichier d'indices séparé est utilisé. v6 reste reproductible.

## Challenge n° 3 — est-ce que ça fait bouger la métrique ?
- Une bonne AUC ne garantit pas un bon mIoU par image : la difficulté est le **seuil par image**, pour des trous
  petits et variables. → La porte de décision porte sur le **mIoU réaliste**, avec un seul seuil relatif réglé hors
  pli, pas sur l'AUC. La borne haute (meilleur seuil par image) sert seulement à jauger le potentiel.
- Sur-ajustement : 170 images seulement. → Règle à 3 paramètres au plus, réglée hors pli à 5 plis. Le U-Net est
  validé hors pli de la même façon.
- Les 10 % d'objets partiels classés « presque tatoués » : une règle **indépendante du régime** (masque = indices
  présents) se corrige d'elle-même, puisque l'objet porte le tatouage et le reste non. → Règle (a) indépendante du
  régime en premier choix.
- Les rectangles à leur propre échelle ne sont pas visibles dans une carte pleine échelle. → On garde la recherche
  de rectangles en priorité sur les images « presque propres ».
- **Bonus** si le temps le permet : le même S1 sur TrustMark (49 % de trous, 0,535), sans retarder PixelSeal.

## Journal d'exécution
- **23:20 — P0** : job « réservoir » 15903731 soumis depuis le dossier de l'équipe (en file, démarrage estimé vers 00:06).
- **23:35 — P1, contrôle synthétique** : S1 (gradient × entrée) et S2 (re-tatouage) donnent une AUC de ~0,5 même sur le
  synthétique. Test élémentaire : le re-tatouage reproduit bien le résidu (corrélation de 0,7 à 0,85), mais le résidu est
  **anti-corrélé au contenu** (−0,10), ce qui noie le signal (+0,03). La variante « message − complément » (S2c)
  échoue aussi : AUC de 0,51 sur les trous et 0,54 sur le synthétique. S3 (canal de détection) : 0,53.
- Les contours des trous réels ne suivent **pas** ceux de l'image (rapport 1,0) ; ceux des objets transplantés, oui
  (1,7 à 2,3). Le synthétique est fidèle sur ces deux points.
- **La porte P1 échoue pour S1, S2 et S3.** La carte actuelle S4 plafonne à 0,51 sur les trous et 0,56 sur les objets
  avec le meilleur seuil par image.
- **Pivot, idée F′ : occlusion.** On mesure la contribution de chaque *zone de l'image* à l'accord avec le message,
  en floutant la zone (σ = 6, la valeur nécessaire pour effacer PixelSeal). Test grossier sur CPU (zones de 64 px,
  3 images par catégorie) : AUC de **0,74 sur les trous, contre 0,64** pour S4. Sur les objets partiels, le signal est
  inversé (accord global ≈ 0), donc on y garde S4, le U-Net et les rectangles.
- **Version retenue** : masques aléatoires couvrant 50 % des cellules de 32 px, ~1 000 essais par image, régression
  ridge de l'accord sur les masques, deux grilles décalées de 16 px. Porte de décision sur les trous : le mIoU réaliste
  hors pli doit battre 0,476 d'au moins +0,04.
- **23:40 — GPU obtenu** (job 15903731, démarré à 23:31). Occlusion zone par zone sur les 83 vrais trous : AUC 0,73-0,74
  contre 0,646 pour S4, mais un mIoU borne haute de seulement 0,534 contre 0,510 : gain réel mais faible. Masques
  aléatoires : AUC 0,53 (50 % masqué) ou 0,69 (15 %, bords doux) : abandonnés.
- **23:45 — Idée décisive : sonde sur le détecteur gelé.** Le ConvNeXt de PixelSeal sait voir son motif. Une petite
  tête de segmentation entraînée sur ses étages 1 à 3 (8, 16 et 32 px) et sur la carte d'accord, **sur synthétique
  seul**, évaluée sur les 170 vraies images non vues : trous 0,476 → 0,575 (AUC 0,81), objets 0,527 → 0,596.
- **23:55 — Construction** : 5 plis avec les vraies images et **epochs fixe** (pas de sélection sur le pli de
  validation), plus un modèle sur tout le train. Hors pli : trous **0,593**, objets **0,629**, PixelSeal global
  0,501 → 0,587 avant rectangles, **~0,62** avec rectangles (contre ~0,55). → **v7**.
- **00:00 — Bonus TrustMark**, même recette sur son ResNet-50, en moyenne 75/25 avec les U-Net : trous 0,531 →
  0,590, objets 0,502 → 0,609, global hors pli 0,565 → 0,619. → **v8**.
- **00:04 — Nœud libéré.** Estimation globale de la v8 : ~0,66 (PixelSeal ~0,62, TrustMark ~0,68, StegaStamp ~0,69),
  contre ~0,62 pour la v6.

# Plan StegaStamp — même recette (22/11, 00:10)
**Objectif** : StegaStamp ~0,69 (U-Net seul) → mieux, avec une sonde sur son décodeur gelé, en moyenne avec les U-Net.
- **S0** : vérifier les jobs de l'équipe, un seul job réservoir (15903990, soumis depuis le dossier de l'équipe).
- **S1** : graphe ONNX → ajouter en sorties `dec_conv_4/5/6/7` (Relu, NCHW : 64@100², 64@50², 128@25², 128@13²), le
  pré-activé de la couche dense et les logits. La copie va dans `~/wm_out` (quota). Accord par position 13×13
  reconstruit par linéarisation des deux couches denses : contribution de chaque position, sous le masque ReLU actif.
- **S2** : sonde = même tête que pour PixelSeal (dimensions 64/64/128) ; synthétique seul d'abord (porte de décision),
  puis 5 plis à epochs fixe + tout le train, et prédictions test.
- **S3** : `ps_final.py --method stegastamp` (moyenne sonde/U-Net choisie hors pli) → v9 = v8 avec StegaStamp
  remplacé. Puis libération du nœud.

**Challenge 1 — validité.** Les caractéristiques d'un décodeur à ReLU et pas de 2 gardent-elles une information
locale ? Le résidu StegaStamp est fort et basse fréquence, donc c'est probable, mais le recalage spatial et les
pas de 2 brouillent un peu la géométrie. Le U-Net fait déjà 0,69, donc le gain relatif sera sans doute plus faible.
→ Porte : la sonde sur synthétique seul doit atteindre au moins le niveau du U-Net sur un sous-type ; sinon,
garder la v8.

**Challenge 2 — faisabilité.** ONNX Runtime avec IO binding et 6 sorties : formes connues, tampons par taille de
lot ; le graphe exécute aussi l'encodeur, ce qui coûte de la mémoire → lots de 32 (64 déjà validé pour les seuls
logits). ~680 images/s → ~25 s par epoch → 5 plis + tout le train ≈ 15 min sur 4 GPU. Le modèle modifié (219 Mo)
va dans `~/wm_out`.

**Challenge 3 — métrique.** Les rectangles StegaStamp ne sont pas gérés par la recherche (désactivée : un décalage
de 4 px suffit à tuer le décodage), donc la sonde doit aussi les couvrir. Évaluation par sous-type, choix par régime
hors pli entre sonde, U-Net et moyenne : on ne remplace que si c'est meilleur.

## Journal StegaStamp
- **00:10** — Graphe ONNX modifié (sorties `dec_conv_4` à `dec_conv_7`, couches denses) dans `~/wm_out/stega/`.
  Accord par position 13×13 obtenu par linéarisation des couches denses : la somme des contributions redonne
  exactement l'accord global (5,392 = 5,392).
- **00:12** — Sonde sur synthétique seul : 0,54 sur les 170 vraies images (trous 0,55, objets 0,60, rectangles 0,47),
  contre ~0,69 pour le U-Net. **Porte non franchie** : le résidu StegaStamp, fort, est déjà bien vu par un U-Net sur
  les pixels.
- **00:16** — 5 plis à epochs fixe + tout le train, puis moyenne avec les U-Net choisie hors pli. Meilleure moyenne :
  0,690 → 0,695 (+0,005 : objets +0,037, rectangles −0,022, trous +0,004). Le gain est **sous la porte de +0,01**,
  partiellement dû à la sélection (75 combinaisons contre 15) et du niveau du bruit (~0,005).
- **Décision : on garde la v8.** Le candidat reste dans `~/wm_out/submissions/v9_candidate.pt`, hors du dossier
  partagé. Nœud 15903990 libéré à 00:19.
