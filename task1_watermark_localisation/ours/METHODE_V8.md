# Méthode V8 → V11 — Localisation de tatouages invisibles (PixelSeal, TrustMark, StegaStamp)

> **Ce document se suffit à lui-même.** Il décrit toute la logique, tous les paramètres et la procédure qui ont mené à
> la soumission **v8** (score estimé ~0,66 en validation honnête), puis à la **v9** (~0,68 ; TTA + décodage calibré, §5.9) à la **v10** (~0,69 ; sonde avec tête dilatée + filtre guidé, §5.10) et à la
> **v11**, qui est la **version recommandée** (~0,695 ; affinage des objets par segments SAM pour PixelSeal et StegaStamp, §5.11). Quelqu'un d'autre doit pouvoir la reproduire à
> partir de ce seul fichier, plus le kit et les données publics des organisateurs.
> Les noms de scripts entre crochets (`[synth.py]`…) renvoient à l'implémentation d'origine. Ils ne sont pas
> nécessaires : chaque brique est spécifiée ici.

---

## 0. Résumé

- **Tâche** : pour 1 020 images test (512×512), prédire un masque binaire 128×128 des zones tatouées. Chaque image a
  été tatouée par **PixelSeal**, **TrustMark** ou **StegaStamp**. Métrique : mIoU par image sur les classes {0, 1},
  puis moyenne sur les images.
- **Atout décisif** : en test, on connaît la **méthode** et le **message** embarqué. Tout le pipeline s'appuie sur ce
  message connu.
- **Pipeline v8**, par image :
  1. **Régime** (« presque tout tatoué » ou « presque rien ») déduit de la bit accuracy du décodage global.
  2. **Masque de base**, selon la méthode :
     - PixelSeal : une **sonde**, c'est-à-dire une petite tête de segmentation entraînée sur les couches
       intermédiaires du **décodeur officiel gelé** ;
     - TrustMark : 75 % sonde + 25 % U-Net ;
     - StegaStamp : ensemble de 6 **U-Net**.
  3. Pour les images partielles de PixelSeal et TrustMark : **recherche du rectangle exact** par décodage du message.
     Si elle aboutit, le masque devient ce rectangle.
- **Scores estimés** :

  | | PixelSeal | TrustMark | StegaStamp | global |
  |---|---|---|---|---|
  | v8 | ~0,62 | ~0,68 | ~0,69 | ~0,66 |
  | v9 (§5.9) | ~0,63 | ~0,69 | ~0,715 | ~0,68 |
  | v10 (§5.10) | ~0,64 | ~0,71 | ~0,715 | ~0,69 |
  | **v11** (§5.11) | ~0,655 | ~0,71 | ~0,72 | **~0,695** |

- **v9 = même cartes de probabilité, meilleure décision**, pour les 3 méthodes :
  - TTA par 4 translations de 4 px ;
  - calibration isotone, modèle par modèle ;
  - décodage qui maximise le mIoU **attendu**, au lieu d'un seuil fixe.
  
  Le gain est validé en validation croisée **imbriquée**, avec un IC 95 % strictement positif pour chaque méthode.

  Références : tout à zéro = 0,25 ; v1 (régime seul) = 0,45 ; v6 = ~0,62. Aucun score du leaderboard n'est connu.

---

## 1. Tâche, données et kit

**Données** (dataset Hugging Face `SprintML/Hackathon_Watermark_Localisation_Task_Paris`, archives `train.zip` et
`test.zip`, extraites dans `release/`) :
- `release/train` : 510 images PNG 512×512 (170 par méthode), masques 512 (`masks/<nom>.png`, 255 = tatoué) et grille
  officielle 128 (`masks_128.npz` : `names`, `masks`).
- `release/test` : 1 020 images (340 par méthode).
- `metadata.json` :
  - `image_name` ;
  - `method` ;
  - `message`, en chaîne de bits : 256 pour PixelSeal, 100 pour TrustMark (variante Q) et StegaStamp ;
  - en train seulement, `full_watermark` (voir les régimes ci-dessous).

**Métrique** :
- Grille 128×128 : une cellule de 4×4 px vaut 1 si au moins 50 % de ses pixels sont tatoués (`downsample_mask`).
- Pour chaque image : (IoU₀ + IoU₁) / 2, avec IoU = 1 si l'union est vide. Score = moyenne sur les images.
- **Chaque image pèse autant** : une petite zone ratée coûte aussi cher qu'une grande.

**Soumission** :
- `torch.save({"image_names": [str]*1020, "masks": bool tensor [1020,128,128]})`, ~16,7 Mo (limite 20-25 Mo).
- Chargée avec `weights_only=True`.
- Envoi par POST sur `BASE_URL/submit/34-watermark-localisation-image`, avec l'en-tête `X-API-Key`.
  `BASE_URL` supposé `http://34.147.229.55` (adresse du leaderboard, non confirmée).

**Kit** (`wmloc/`, dans le même dataset) :
- `load_watermark(m, models_dir, device)` renvoie un wrapper avec `embed(img_uint8, bits)`, `logits(B×3×H×W in [0,1])`,
  `decode` et `features`.
- Chaque décodeur redimensionne l'entrée à sa propre résolution :
  - PixelSeal 256 (bilinéaire, antialias) ;
  - TrustMark 245 (bilinéaire, antialias, puis ×2−1) ;
  - StegaStamp 400 (bicubique, antialias, clamp).
- `features()` de PixelSeal et de TrustMark expose `per_position_logits` (grille 8×8).
- `utils` fournit `downsample_mask` et `miou_image`.
- Les modèles se téléchargent avec `download_models.py` :
  - dépôt `facebookresearch/videoseal` au commit `b1bf4c0` ;
  - checkpoint PixelSeal `https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth` ;
  - ONNX StegaStamp de WAVES, auquel le script ajoute la sortie `gen_decoder/decoder_out/BiasAdd:0` (logits).

**Deux régimes** :
- `full_watermark=True` : l'image entière a été tatouée, puis un « trou » a été restauré à l'original propre, ou
  remplacé par un objet étranger. Fraction tatouée de 0,51 à 0,997.
- `full_watermark=False` : seule une petite zone est tatouée (~9 % en moyenne, de 0,003 à 0,44). C'est un **rectangle**
  ou un **objet**.

---

## 2. Mise en place

### 2.1 Logiciels et ressources (testé)
- **Python 3.12.3**, avec : torch 2.5.1, torchvision 0.20.1, numpy 1.26.4, scipy 1.17.1, onnx 1.23.2,
  onnxruntime-gpu 1.23.2, trustmark 0.8.0, timm 1.0.31, opencv-python 4.11, pycocotools 2.0.11, pillow, omegaconf,
  einops, safetensors.
- **GPU** : A100 40 Go. Un nœud de 4 GPU suffit largement.

### 2.2 Téléchargements préalables (les nœuds de calcul n'ont pas internet)
**Poids TrustMark Q**
- Téléchargés à la main, car `trustmark` le fait d'habitude au premier usage.
- Source : `https://cc-assets.netlify.app/watermarking/trustmark-models/<fichier>`.
- Destination : `site-packages/trustmark/models/`.

| fichier | md5 |
|---|---|
| `trustmark_Q.yaml` | `fe40df84a7feeebfceb7a7678d7e6ec6` |
| `decoder_Q.ckpt` | `4ced90e9cfe13e3295ad082887fe9187` |
| `encoder_Q.ckpt` | `700328b8754db934b2f6cb5e5185d81f` |
| `trustmark_rm_Q.yaml` | `8476bcd4092abf302272868f3b4c2e38` |
| `trustmark_rm_Q.ckpt` | `760337a5596e665aed2ab5c49aa5284f` |

**COCO**
- `http://images.cocodataset.org/zips/val2017.zip` (5 000 images).
- `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` : seul `instances_val2017.json` sert.
- Placé dans un dossier `coco/`, à côté du dossier de la tâche.

**timm**
- `resnet34.a1_in1k`, téléchargé une fois avec `HF_HOME` pointé vers un cache local, puis utilisé avec
  `HF_HUB_OFFLINE=1`.

**SAM (à partir de la v11)**
- `pip install --no-deps --target <dossier> segment-anything` (version 1.0), puis ajout du dossier au `sys.path`,
  sans toucher au venv d'équipe.
- Poids ViT-B : `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` (md5
  `01ec64d29a2fca3f0661936605ae66f8`).

### 2.3 Environnement JURECA (si l'on réutilise le même)
**Accès**
- Hôte SSH `juelich` (jureca.fz-juelich.de, utilisateur `vaudois1`), avec **2FA**. `~/.ssh/config` contient
  `ControlMaster auto`, `ControlPath ~/.ssh/cm-%r@%h:%p` et `ControlPersist 12h`.
- Si `ssh -O check juelich` échoue, il faut lancer soi-même `ssh-add ~/.ssh/id_ed25519` puis
  `ssh -fN juelich`, et saisir son code OTP.

**Dossiers**
- Dossier de l'équipe : `/p/scratch/training2666/denoncin1/Ca hack les hack/`.
- Dossier de la tâche : sous-dossier `Hackathon_Watermark_Localisation_Task/` (venv d'équipe `venv/`, `models/`,
  `release/`, `wmloc/`). COCO est dans `../coco`.
- Notre code est dans le sous-dossier `ours/`, monté en local par sshfs dans `~/tek5/hackaton/wm`.

**Python**
- Le venv a été créé avec le module Python de JURECA. Sans
  `module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3`, Python **segfault**, car il charge la `libpython` du
  système.
- Toujours lancer via un petit script qui fait : `module load …` ; `cd ours` ;
  `export CUDA_VISIBLE_DEVICES=$GPU` (choix du GPU) ; `export HF_HUB_OFFLINE=1 HF_HOME=../.cache/huggingface
  OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1` ; `exec ../venv/bin/python "$@"`.

**Slurm** : compte `training2666`, réservation `cispahack`, partition `dc-gpu`. Un job reçoit toujours un **nœud
entier** (4×A100, 256 CPU), même avec `--gres=gpu:1`.

**Règle d'équipe (impérative)**
- La réservation est pleine. **Toute l'équipe** (tous les membres du dossier de l'équipe) n'a droit qu'à **un seul
  job à la fois**, en cours ou en file.
- Avant toute soumission :
  ```bash
  squeue -r --reservation=cispahack -h -o "%u|%T|%Z" | grep "<dossier équipe>"
  ```
  Si la commande renvoie quelque chose, on ne soumet pas.
- Soumettre **depuis le dossier de l'équipe** : la vérification filtre sur le dossier de travail du job.
- Besoin de plusieurs GPU : un seul job, parallélisé en interne.
- Travail supplémentaire : l'enchaîner avec `--dependency=afterany:<id>`, ne jamais empiler les jobs.

**Façon de travailler éprouvée**
- Un job « réservoir » `sbatch` (4 GPU, `--cpus-per-task=64`, 2 h 30, contenu : `sleep 9000`) soumis depuis le dossier
  de l'équipe.
- Toutes les tâches s'y exécutent en étapes :
  ```bash
  GPU=k srun --jobid=<id> --overlap -n1 -c30 --export=ALL <lanceur> script ...
  ```
  - **Toujours `-n1`** : sinon `srun` lance plusieurs copies.
  - Lancer en arrière-plan (`nohup … &`) pour occuper les 4 GPU.
- `scancel <id>` dès que le travail est fini.

**Quota**
- Le quota de fichiers et d'octets du scratch est **partagé par tout le projet** et sature par intermittence
  (`Errno 122`).
- Donc :
  - écrire les sorties précieuses dans le **home** (`~/wm_out/`, quota séparé de 20 Go) : soumissions, résultats de
    recherche, sondes, logs ;
  - stocker le synthétique en **shards** de 250 images, jamais un fichier par image ;
  - écrire de façon atomique (fichier temporaire, puis renommage) et prévoir une **reprise** (sauter ce qui est
    déjà fait).

**Pièges**
- `pkill -f motif` tue aussi la session SSH si le motif figure dans sa propre ligne de commande. Utiliser
  `scancel <job.étape>` ou `pkill -f "[g]lobal_rect…"`.
- Dans `ssh '… cd X && cmd & …'`, le `cd` part dans un sous-shell. Écrire `cd X || exit; cmd &`.
- Ne pas entourer `salloc --no-shell` d'un `timeout` : le job est annulé.
- **Calculs lourds uniquement dans le job Slurm** (`srun --overlap`), jamais longtemps sur le nœud de login. Un
  lancement de SAM sur CPU (~25 min, 16 threads) n'y a pas été autorisé.

### 2.4 Conventions de l'équipe
- Les soumissions sont faites **à la main** ; la clé API n'est jamais écrite dans le code. Le `submission_template.py` du kit a été **modifié** en
  conséquence :
  - `BASE_URL` = variable `WM_BASE_URL`, par défaut `http://34.147.229.55` ;
  - `API_KEY` = variable `WM_API_KEY`, jamais écrite dans le code ;
  - option `--submit <fichier>`, qui appelle seulement la fonction `submit(fichier)` du kit puis s'arrête ;
  - message d'erreur clair si la clé est absente.

  Commande type :
  ```bash
  read -s -p "API key: " WM_API_KEY; export WM_API_KEY
  venv/bin/python submission_template.py --submit ours/submissions/v8.pt
  ```
  (sur le nœud de login, après le `module load`, **sans** `srun`).
- Avant un gros chantier, un **plan challengé 3 fois** : validité du signal, faisabilité, effet réel sur la
  métrique.
- Libérer le nœud dès que possible.

---

## 3. Faits établis sur les données (ils justifient chaque brique)

| # | fait | mesure | conséquence |
|---|---|---|---|
| F1 | Les deux régimes sont séparés par la **bit accuracy globale** (décodage de l'image entière, comparé au message connu) : presque tatoué ≈ 0,97-1,0 ; partiel ≈ 0,5 | Seuils appris ; 97,6 / 98,2 / 98,8 % de bonnes classifications | Prédire tout-1 ou tout-0 selon le régime vaut déjà **0,45**, contre 0,25 |
| F2 | Les zones partielles sont à ~50 % des **rectangles** (remplissage de la boîte > 0,95) et à ~50 % des **objets** | Remplissage de la boîte englobante du masque | Deux mécanismes distincts |
| F3 | Les **rectangles** sont tatoués **à leur propre échelle** (le recadrage a été tatoué seul) : décoder **exactement** ce recadrage retrouve le message | Décodage de recadrages élargis d'un facteur s autour de la vérité : l'optimum est à s = 1 ; acc ≥ 0,9 pour 23 à 44 % des images partielles | → **recherche du rectangle exact** |
| F4 | Les **objets** portent le tatouage **de l'image entière, à la même position** (image tatouée à 512, objet collé sur place) | La carte d'accord est forte sur les objets réels (PixelSeal 1,13 contre 0,53 hors objet), nulle sur les rectangles. Parmi 4 variantes synthétiques, seule « tatouage pleine image, objet gardé sur place » reproduit ces valeurs ; déplacer l'objet tue le signal, car la phase des pixels change | Le générateur synthétique colle l'objet **sur place** |
| F5 | Le contour des **trous** est **invisible** dans l'image ; celui des **objets transplantés** est visible | Gradient moyen sur le contour vrai / sur ce contour décalé de 24 px : trous ≈ 1,0 ; objets 1,7 à 2,3 | Pour les trous, seul l'indice de tatouage aide |
| F6 | **StegaStamp** : décaler la boîte de 4 px fait chuter l'accord de 0,81 à 0,55 (vérifié aussi à 8 et 16 px) | Décodage de la boîte vraie, puis décalée | Recherche de rectangle impossible pour StegaStamp |
| F7 | **PixelSeal** : résidu = 0,2 × JND(x) × agrandi×2(UNet(luminance à 256, message)). Il est gris (R = G = B), lisse à 2 px, d'amplitude ~1-2 niveaux. Le détecteur est un ConvNeXt-tiny à 256, de sortie 8×8 (`pixelwise: False`) | Code de `videoseal` (embedder `unet_base_yuv_quant`, atténuation `jnd_1_1`, `scaling_w` 0,2) | Les cartes natives 8×8 sont trop grossières pour des trous de ~10 % |
| F8 | Le résidu PixelSeal re-tatoué est reproductible (corrélation 0,7-0,85), mais **anti-corrélé au contenu** (−0,10) | Tatouage en flottant d'images COCO | Les filtres adaptés au pixel échouent (§10) |
| F9 | Bilan des pertes PixelSeal (v6) : trous 49 % des images (perte 0,256), objets 25 % (0,117), rectangles 25 % (0,132) | mIoU par sous-type | Priorité aux trous et aux objets |

**Principe** : on se sert des **décodeurs officiels** et du message connu comme d'**instruments de mesure** pour
reconstituer la façon dont les données ont été générées. On corrige ensuite le générateur synthétique jusqu'à ce que
ses statistiques collent au réel.

---

## 4. Vue d'ensemble de la v8

```
image test (méthode m, message s)
 1. régime    : bitacc(décodage global) ≥ seuil_m  →  full / partiel
 2. base      : PixelSeal  → proba = sonde (moyenne de 6 têtes)
                TrustMark  → proba = 0,75·sonde + 0,25·U-Net (6 têtes / 6 U-Net)
                StegaStamp → proba = U-Net (moyenne de 6)
                masque = post-traitement(proba, seuil, nettoyage) selon (m, régime)
 3. rectangle : si régime partiel et m ∈ {PixelSeal, TrustMark} et rectangle trouvé avec bitacc ≥ 0,8 → masque = rectangle
```

> **En v9** (§5.9), seule l'étape 2 change. Les mêmes probabilités sont calculées avec une TTA à 4 translations ;
> chaque modèle est calibré (isotone, par régime), puis les modèles sont moyennés. Le masque est ensuite obtenu par
> **décodage du mIoU attendu** (s = 1, c = 16 ; StegaStamp partiel : s = 0, c = 0) au lieu du seuil fixe du tableau
> ci-dessous. Les étapes 1 et 3 sont identiques.

**Réglages finaux de la v8** (tous choisis hors pli, §6) :

| méthode | régime | probabilité | seuil | taille min. des composantes (cellules) | rectangles acceptés en test |
|---|---|---|---|---|---|
| PixelSeal | presque tatoué | sonde | 0,7 | 16 (sur les trous) | — |
| PixelSeal | partiel | sonde | 0,3 | 16 | 38 / 164 |
| TrustMark | presque tatoué | 0,75 sonde + 0,25 U-Net | 0,7 | 0 | — |
| TrustMark | partiel | 0,75 sonde + 0,25 U-Net | 0,3 | 16 | 79 / 168 |
| StegaStamp | presque tatoué | U-Net | 0,7 | 16 (sur les trous) | — |
| StegaStamp | partiel | U-Net | 0,3 | 0 | désactivé |

**Post-traitement** :
- masque = (proba > seuil) ;
- régime presque tatoué : retirer les composantes connexes de **trou** (classe 0) de moins de *c* cellules ;
- régime partiel : retirer les composantes **tatouées** de moins de *c* cellules ;
- étiquetage avec `scipy.ndimage.label` (connexité 4).

---

## 5. Les briques (spécification complète)

### 5.1 Classifieur de régime [global_logits.py, regime.py]
1. Pour chaque image (train et test) : logits du décodeur sur l'image entière (wrapper du kit), multipliés par
   s = 2·message − 1.
2. bitacc = proportion de logits signés > 0 ; on garde aussi la **moyenne des logits signés**.
3. Seuil par méthode, sur les 170 images train :
   - prendre les valeurs de bitacc triées comme seuils candidats ;
   - garder celui qui maximise l'exactitude de « bitacc ≥ t ⇔ `full_watermark` » ;
   - le seuil final est le **milieu** entre t et la plus grande valeur strictement inférieure.
4. Valeurs obtenues : **PixelSeal 0,798828125 ; TrustMark 0,78 ; StegaStamp 0,755**.
5. `is_full(x) = bitacc(x) ≥ seuil_méthode`.

### 5.2 Cartes d'accord avec le message (PixelSeal, TrustMark) [maps.py, feats.py]
- `per_position_logits` (grille 8×8, `nbits` canaux) donne une carte = moyenne sur k de s_k · logit_k(position).
- **Shift-and-stitch** :
  - 16 versions de l'image (dy, dx ∈ {−24, −8, +8, +24}) ;
  - le contenu est translaté pour que le pixel (y, x) de l'original arrive en (y−dy, x−dx), avec un bord en miroir de
    32 px ;
  - la carte 8×8 du décalage (a, b) remplit `H[a::4, b::4]` d'une grille 32×32 ;
  - agrandissement bilinéaire à 128×128.
- **Canaux auxiliaires** du U-Net (128×128) :
  - [carte/5 ; moyenne des logits signés/5 (constante) ; bitacc·2−1 (constante)] ;
  - la carte vaut 0 pour StegaStamp ;
  - calculés pour toutes les images, réelles et synthétiques.
- La règle sur carte des anciennes versions (seuil relatif par régime) a servi jusqu'à la v6 pour PixelSeal ; elle
  n'est plus utilisée en v8.

### 5.3 Générateur synthétique fidèle « synth2 » [synth.py]
**Ingrédients**
- Image COCO val2017 au hasard → recadrage carré aléatoire de côté min(H, W) → bicubique 512.
- Message aléatoire, **sauvegardé**.
- Tatouage avec les **vrais embedders** du kit : `embed(img_uint8, bits)`, à la taille de l'image fournie.
- **Masque d'objet** :
  - annotation COCO aléatoire (hors `iscrowd`, aire > 400) ;
  - décodée avec pycocotools et recadrée à sa boîte ;
  - redimensionnée en bilinéaire, puis seuillée à > 127.

**Tirages**
- `aire` = exp(U(ln 0,004 ; ln 0,45)), en fraction de 512².
- **Rectangle** d'aire `a` (en px²) :
  - proportion r = exp(U(0, ln MAXR)) avec probabilité 0,8, sinon exp(U(0, ln min(MAXR, 1,5))) ;
  - r inversé avec probabilité 0,5 ;
  - h = clip(√(a/r), 24, 512), w = clip(a/h, 24, 512), position uniforme ;
  - MAXR = 16 pour PixelSeal et StegaStamp, 2 pour TrustMark.
- **Objet placé** à une aire cible : facteur d'échelle = √(aire·512² / (remplissage × aire de sa boîte)), côtés
  bornés à [8, 512], position uniforme.

**Échantillon** : régime presque tatoué ou partiel avec probabilité 0,5 chacun.
- **Partiel** (rectangle tiré à partir d'une aire)
  - 50 % **« rectangle à son échelle »** (F3) :
    - source = le recadrage de l'image elle-même (50 %), ou le coin h×w d'une autre image COCO (50 %) ;
    - sortie[rect] = `embed(source)` ;
    - masque = rectangle.
  - 50 % **« transplantation sur place »** (F4) :
    - donneuse = une autre image COCO (60 %) ou l'image elle-même (40 %), **tatouée en entier à 512** ;
    - un masque d'objet COCO est redimensionné dans le rectangle (rectangle entier s'il fait < 30 px) ;
    - sortie[masque] = donneuse_tatouée[masque], **mêmes coordonnées**.
- **Presque tatoué**
  - image entière tatouée ;
  - trou = union de 1 objet (70 %) ou de 2-3 objets (30 %), placés avec une aire de `aire`×0,9 ;
  - si la fraction du trou dépasse 0,49 ou fait moins de 20 px : 1 objet d'aire 0,05 ;
  - contenu du trou : l'original propre (60 %) ou les pixels d'une autre image COCO (40 %) ;
  - masque = non-trou.

**Masque et stockage**
- Masque 128 obtenu par le `downsample_mask` officiel.
- Shards de 250 images : `.npy` uint8 (N×512×512×3) + `.npz` (`names`, `mask128`, `full`, `message`).
- Volume : graines 21 et 22, 3 500 images chacune, soit **7 000 par méthode**.

**Contrôle de réalisme** (à refaire si l'on modifie le générateur) : comparer réel et synthétique sur
- la bit accuracy et le logit signé moyen par régime ;
- la carte d'accord moyenne sur les zones tatouées et propres, par régime et par forme ;
- le rapport de contour (F5).

Le générateur v1 (objets **déplacés**) donnait une carte nulle sur les objets synthétiques, alors qu'elle est forte
sur le réel : c'était faux.

### 5.4 U-Net (utilisé pour StegaStamp, et à 25 % pour TrustMark) [train_seg.py]
**Architecture**
- Entrée à 6 canaux : RGB normalisé ImageNet + passe-haut ×10, avec passe-haut = x − moyenne 5×5 (bord en miroir).
- Encodeur `timm resnet34.a1_in1k`, `features_only`, `in_chans=6` (pas de 2, 4, 8, 16 et 32).
- Chemin pleine résolution : conv3×3(6→32)-GELU, conv3×3 de pas 2 (32)-GELU, conv3×3 de pas 2 (32)-GELU, ce qui donne
  le pas 4.
- Décodeur : on part du pas 32. Pour les sauts aux pas 16, 8 et 4 :
  - agrandissement bilinéaire ;
  - concaténation [y, saut, canaux auxiliaires redimensionnés] ;
  - 2 × (conv3×3 → 128, BN, GELU).
- Tête : [y, chemin pleine résolution, canaux auxiliaires] → conv3×3 → 64, GELU → conv1×1 → 1.
- **Sortie 128×128 directe** (pas 4 = grille officielle).

**Perte** : BCE + (1 − mIoU souple). mIoU souple par image = moyenne sur les classes {1, 0} de
(Σ p·g + 1) / (Σ (p + g − p·g) + 1), avec p = sigmoïde.

**Entraînement**
- Données : synth2 (7 000) + vraies images du train (hors pli de validation) **répétées ×4**.
- AdamW (lr 3e-4, wd 1e-4), OneCycle (pct_start 0,05), 10 epochs, lots de 16, bf16.
- **Aucune augmentation** : retournements et changements d'échelle détruiraient le motif.
- ~40 s par epoch seul sur un A100.

**Plis** (communs à tout le pipeline) : pour chaque méthode, trier les 170 images train par nom ; dans chaque groupe
`full_watermark` ∈ {False, True}, le j-ième reçoit le pli j mod 5.

**Modèles**
- 5 modèles de pli (`f0` à `f4`) : on garde l'epoch au meilleur mIoU de validation (seuil 0,5). Leurs
  probabilités de validation forment l'**OOF**.
- Un modèle `f-1` sur tout le train (dernier epoch).
- En test : moyenne des probabilités des 6.

### 5.5 Choix du prédicteur de base par régime [oof.py]
- Pour chaque (méthode, régime prédit), on compare sur les probabilités OOF :
  - la constante (masque plein ou vide selon le régime) ;
  - la règle sur carte ;
  - « U-Net, seuil t, nettoyage c », avec t ∈ {0,3 ; 0,4 ; 0,5 ; 0,6 ; 0,7} et c ∈ {0 ; 4 ; 16}.
- On garde le meilleur mIoU moyen.
- Résultat retenu pour StegaStamp : partiel `U-Net@0,3/min0` (0,635), presque tatoué `U-Net@0,7/min16` (0,744).

### 5.6 Recherche du rectangle exact [global_rect.py, rect_search.py]
**Périmètre** : uniquement les images test dont le régime prédit est **partiel**, pour TrustMark et PixelSeal.

**Évaluation exacte d'une boîte**
- Recadrer l'image 512 sur la boîte.
- Redimensionner à la résolution du décodeur, avec antialias : bilinéaire 256 pour PixelSeal, bilinéaire 245 pour
  TrustMark, bicubique 400 + clamp pour StegaStamp.
- Décoder, puis calculer le **logit signé moyen** (score) et la bitacc.

**Candidats**
- Tailles h, w ∈ {28, 40, 56, 80, 112, 160, 224, 320, 448, 512}, avec max/min ≤ 2 (TrustMark) ou ≤ 16 (PixelSeal).
- Pas = max(8, côté // 3) sur chaque axe, en ajoutant toujours la position collée au bord.
- Soit ~19 200 boîtes pour TrustMark et ~28 500 pour PixelSeal.

**Étape 1**
- Toutes les boîtes sont recadrées et redimensionnées en un seul appel `torchvision.ops.roi_align` (sortie à la
  résolution du décodeur, `sampling_ratio=0`, `aligned=True`), décodées par lots en bf16.
- Score = logit signé moyen.

**Étape 2**
- 200 meilleures boîtes, suppression des doublons (IoU < 0,5), on en garde **6**.
- Chacune est affinée par **hill-climbing des 4 bords** :
  - pas de 16, puis 8, puis 4, puis 2 px, jusqu'à 12 itérations par pas ;
  - à chaque itération, on essaie chaque bord ±pas (côté minimal 16 px, boîte bornée à l'image) avec l'évaluation
    exacte ;
  - on prend le meilleur mouvement s'il augmente le score, sinon on passe au pas suivant.
- On garde la meilleure boîte finale.

**Acceptation** si **bitacc ≥ 0,8**. Le masque devient alors le rectangle, ramené en 128 par `downsample_mask`.

**Validation et durées**
- TrustMark : 100 % des vrais rectangles retrouvés (mIoU 0,93), **0 fausse détection** à 0,8 (contre 1 sur 17 à 0,75).
- PixelSeal : ~44 % des rectangles retrouvés (mIoU 0,94), 0 fausse détection à 0,8 (mais 1-2 sur 17 à 0,7).
  C'est le plafond : les petits ou fins rectangles ne se décodent pas, même avec la boîte exacte.
- Durées sur A100 : ~6 s par image pour TrustMark, ~26 s pour PixelSeal.
- StegaStamp : désactivé (F6).

### 5.7 Sonde sur décodeur gelé (le levier de la v7 et de la v8) [ps_probe.py]
**Idée** : le décodeur officiel a appris à reconnaître son propre motif. Ses **couches intermédiaires**, plus fines que
sa sortie, portent une information **locale** de présence du tatouage. On **gèle** le décodeur (mode eval, sans
gradient) et on entraîne seulement une petite tête.

**Backbone PixelSeal** (détecteur du wrapper)
- Image 512 → redimensionnement du kit à 256 → normalisation (x·2−1) → ConvNeXt-tiny (profondeurs 3, 3, 9, 3 ;
  largeurs 96, 192, 384, 768).
- Pour i = 0 à 3 : couche de sous-échantillonnage i, puis étage i. On garde :
  - **f1** = étage 1 (96×64×64) ;
  - **f2** = étage 2 (192×32×32) ;
  - **f3** = étage 3 (384×16×16) ;
  - f4 = étage 4 (768×8×8).
- `pixel_decoder.output_upscaling(f4)`, puis la couche linéaire appliquée **à chaque position** : 257 canaux en 8×8.
  - canal 0 = **détection** (`det0`) ;
  - canaux 1 à 256 = bits ;
  - **accord** = moyenne sur k de s_k · canal_(k+1).

**Backbone TrustMark** (ResNet-50 du décodeur)
- Préparation du kit : 245, bilinéaire antialias, ×2−1 → conv1, bn1, relu, maxpool.
- **f1** = layer1 (256×62×62), **f2** = layer2 (512×31×31), **f3** = layer3 (1024×16×16).
- La couche `fc` est appliquée à chaque position de layer4 (2048×8×8) : 100 canaux.
- **Accord** = moyenne sur k de s_k · canal_k ; `det0` = 0.

**Tête** (identique pour les deux méthodes)
- f1, f2 et f3 passent chacun par conv1×1 → 64 + BN, puis sont ramenés à la taille de f1 (bilinéaire).
- Concaténation avec :
  - l'accord agrandi / 5 ;
  - l'**accord global** (moyenne spatiale, diffusé partout) / 5 ;
  - `det0` agrandi / 5.
- Soit 195 canaux → 4 × [conv3×3, BN, GELU] de largeurs 128, 128, 128, 64 → conv1×1 → 1 → agrandissement
  bilinéaire à 128×128 (logits).

**Entraînement**
- Données : synth2 de la méthode (7 000) + vraies images du train hors pli **répétées ×8**.
- Perte BCE + (1 − mIoU souple).
- AdamW (lr 1e-3, wd 1e-4), OneCycle (pct_start 0,1), lots de 32, bf16, **5 epochs, on garde le dernier** (aucune
  sélection sur le pli de validation).
- ~14 s par epoch.

**Modèles**
- 5 plis (mêmes plis qu'en §5.4) → probabilités OOF honnêtes.
- 1 modèle sur tout le train (5 epochs).
- Pour chacun : probabilités test. **Probabilité test = moyenne des 6.**

**Porte de décision avant d'investir** : une sonde entraînée sur **synthétique seul** doit battre la référence sur les
**vraies** images, toutes non vues. Pour PixelSeal, c'était déjà le cas : trous 0,476 → 0,575 (AUC 0,81), objets
0,527 → 0,596.

**Intégration** [ps_final.py]
- Pour chaque régime prédit, on choisit sur l'OOF (t, c, b) :
  - t ∈ {0,3…0,7}, c ∈ {0, 4, 16} ;
  - proba = b·sonde + (1−b)·U-Net (OOF des U-Net de §5.4), avec b ∈ {1 ; 0,75 ; 0,5}.
- PixelSeal : sonde seule (b = 1). TrustMark : b = 0,75 retenu dans les deux régimes.
- Ensuite, le remplacement par les rectangles s'applique (§5.6).

### 5.8 Assemblage des versions
- **v6** = choix de §5.5 par méthode et régime (PixelSeal : règle sur carte ; TrustMark et StegaStamp : U-Net) +
  rectangles.
- **v7** = v6 dont les 340 masques PixelSeal sont recalculés avec la sonde + les rectangles.
- **v8** = v7 dont les masques TrustMark sont recalculés avec 0,75 sonde + 0,25 U-Net + les rectangles.

**Contrôles systématiques** avant de livrer :
- 1 020 noms uniques, dans l'ordre de `metadata.json` du test ;
- tenseur bool [1020, 128, 128] ;
- rechargement avec `weights_only=True` ;
- ~16,7 Mo ;
- fraction de 1 par méthode autour de 0,48-0,50 (les régimes sont ~50/50).

### 5.9 v9 : TTA + calibration + décodage « mIoU attendu » [tta_infer.py, decision.py, tta_eval.py, final_v9.py]
**Pourquoi.** Après la v8, le classement des pixels est bon, mais la décision est mauvaise. Exemple : StegaStamp a une
AUC de 0,90 à 0,95 mais un mIoU de 0,63 à 0,74, et sous-segmente les rectangles (2,6 % prédits contre 5 % réels). Le
meilleur seuil choisi image par image (borne haute) donnerait +0,033 (PixelSeal), +0,035 (TrustMark) et +0,050
(StegaStamp).

**Les mêmes modèles qu'en v8** (aucun ré-entraînement) :
- PixelSeal : sonde (6 têtes) ;
- TrustMark : 0,75 sonde + 0,25 U-Net, mélangés **modèle par modèle** (tête du pli k avec U-Net du pli k, « tout le
  train » avec « tout le train ») ;
- StegaStamp : U-Net (6).

**1. TTA**
- 4 décalages (dy, dx) ∈ {(0,0), (4,0), (0,4), (4,4)} px, soit des multiples de 4 px, c'est-à-dire **une cellule
  entière** de la grille 128.
- L'image est translatée avec un bord en miroir de 32 px, de sorte que x_s[y, x] = x[y+dy, x+dx]. Les cartes
  auxiliaires 128 du U-Net sont translatées de (dy/4, dx/4) cellules, avec bord répliqué.
- La sortie est recalée **exactement** à l'inverse, avec bord répliqué.
- On moyenne les 4 probabilités.
- Calculé pour chaque modèle : sur son pli de validation (hors pli) et sur le test.

**2. Calibration isotone**, par méthode et par **régime prédit**
- Ajustée sur les cartes hors pli avec TTA (pour l'évaluation imbriquée : sur les 4 autres plis seulement).
- Construction :
  - 100 classes de quantiles de p ;
  - fraction de pixels vraiment tatoués par classe ;
  - pool-adjacent-violators pour imposer la monotonie ;
  - interpolation linéaire entre les centres des classes.
- **En test**, chaque modèle est calibré **séparément** par la courbe de sa famille et de son régime, puis les 6
  modèles sont **moyennés**.

**3. Décodage qui maximise le mIoU attendu**, indépendant du régime et sans seuil
- q = carte calibrée, éventuellement lissée par une gaussienne de σ = s cellule(s).
- Trier les N = 16 384 cellules par q décroissant. S_K = somme des K plus grandes valeurs, T = Σq.
- Pour K = 0..N :
  E[mIoU](K) ≈ ½·[ S_K / (K + T − S_K) + (N − K − T + S_K) / (N − S_K) ],
  en prenant 1 quand un dénominateur est nul.
- Garder les K* cellules qui maximisent cette quantité.
- Puis nettoyage c : si le masque couvre ≥ 50 % de l'image, retirer les trous (composantes de classe 0) de moins de
  c cellules ; sinon, les composantes tatouées de moins de c cellules.

**4. Réglages (s, c)**, choisis par régime sur les cartes hors pli : partout **s = 1, c = 16**, sauf StegaStamp
partiel (**s = 0, c = 0**).

**5. Ensuite**, le remplacement par les rectangles est inchangé (PixelSeal et TrustMark, acc ≥ 0,8 : 38 et 79
images).

**Validation imbriquée** (réglages choisis sur 4 plis, score sur le 5ᵉ), gain apparié contre la décision v8
re-sélectionnée de la même façon, avec IC bootstrap 95 % :

| méthode | TTA seule | décodage seul | **TTA + décodage (v9)** | IC 95 % |
|---|---|---|---|---|
| StegaStamp | +0,003 | +0,021 | **+0,026** (0,690 → 0,715) | +0,016 à +0,036 |
| TrustMark | +0,004 | +0,005 | **+0,008** (0,577 → 0,585) | +0,003 à +0,013 |
| PixelSeal | +0,005 | +0,003 | **+0,009** (0,574 → 0,583) | +0,003 à +0,015 |

Ces chiffres sont **sans** les rectangles. Les gains viennent surtout des trous et des objets, que la recherche de
rectangles ne touche pas.

**Contrôles de cohérence** (fraction tatouée prédite, hors pli → test) :
- PixelSeal : presque tatoué 0,876 → 0,878 ; partiel 0,062 → 0,067.
- TrustMark : 0,915 → 0,883 ; 0,101 → 0,092.
- StegaStamp : 0,915 → 0,873 ; 0,073 → 0,077.

La baisse en test du régime presque tatoué pour TrustMark et StegaStamp n'est **pas** un artefact de la moyenne des
modèles. La v8 montre la même (0,885 et 0,870), et le décodage modèle par modèle donne les mêmes valeurs. C'est une
propriété des images test.

**v9 contre v8** : 2,8 % (PixelSeal), 2,3 % (TrustMark) et 1,7 % (StegaStamp) des cellules changent.

### 5.10 v10 : sonde avec tête dilatée + filtre guidé (PixelSeal, TrustMark) [ps_probe.py --dilated, guided.py, eval_v10.py, final_v10.py]
**Pourquoi.** Après la v9, la décision est presque à son plafond (meilleur seuil par image : seulement +0,02 à +0,04).
Il faut donc de **meilleures cartes**. Pour PixelSeal et TrustMark, les trous restent le premier poste de perte
(~0,19 par méthode). Leur score selon la taille :
- < 2 % : ≈ 0,50 ;
- 5-20 % : 0,64-0,70 ;
- > 20 % : il **redescend** à 0,62-0,64.

La tête de sonde (4 conv3×3 sur la grille 64) ne voit que ~70 px : au centre d'un grand trou, elle n'a aucun contexte.

**1. Tête dilatée** (`Head(dilated=True)`)
- Même tête que §5.7, mais les 4 blocs conv3×3 ont des **dilatations 1, 2, 4, 8** (padding = dilatation), soit un
  champ récepteur de ~31 cellules (~250 px).
- Entraînement identique (synth2 + vraies images ×8, AdamW 1e-3, OneCycle, lots de 32, bf16), mais **8 epochs
  fixes** au lieu de 5.
- 5 plis + tout le train, pour PixelSeal comme pour TrustMark.

**2. Moyenne « fx+dil »** : l'unité du pli k est la moyenne de la tête v9 et de la tête dilatée du pli k (de même pour
« tout le train »). Pour TrustMark, elle est ensuite mélangée à 0,75 avec le U-Net (0,25), comme en v9. TTA, calibration
par unité et par régime, puis moyenne des 6 unités : comme en §5.9.

**3. Filtre guidé** (He et al.), appliqué à la carte calibrée **avant** le lissage et le décodage
- Guide I = luminance de l'image (moyenne RGB / 255) ramenée à 128×128 par moyenne de blocs 4×4.
- Filtres moyens de rayon r (`uniform_filter`, taille 2r+1, bord en miroir).
- Calcul :
  - a = cov(I, q) / (var(I) + ε) ;
  - b = moyenne(q) − a·moyenne(I) ;
  - q' = moyenne(a)·I + moyenne(b), borné à [0, 1].
- Choix par régime (r ∈ {2, 4, 8}, ε ∈ {1e-3, 1e-2}, ou « off »), avec le lissage s et le nettoyage c.
- Retenu :

  | méthode | régime | (r, ε) | s | c |
  |---|---|---|---|---|
  | PixelSeal | presque tatoué | (4 ; 0,01) | 1 | 16 |
  | PixelSeal | partiel | (8 ; 0,001) | 1 | 16 |
  | TrustMark | presque tatoué | (8 ; 0,01) | 1 | 16 |
  | TrustMark | partiel | (8 ; 0,001) | 0 | 0 |

- Pour StegaStamp, le filtre est toujours « off ».
- Le filtre aide même sur les trous (+0,006 à +0,016), qui n'ont pourtant pas de contour visible : il agit comme un
  lissage qui épouse la structure de l'image.

**Validation** (imbriquée, contre la v9 ; **la variante elle-même est choisie dans la validation imbriquée**, parmi 9 :
v9 / dilatée / 3 graines / v9 + dilatée / les 4, chacune avec ou sans filtre) :

| méthode | variante choisie dans les 5 plis | gain | IC 95 % | détail |
|---|---|---|---|---|
| PixelSeal | v9 + dilatée + filtre | **+0,012** (0,583 → 0,595) | +0,007 à +0,018 | trous +0,011, objets +0,028 |
| TrustMark | idem | **+0,0175** (0,585 → 0,603) | +0,012 à +0,024 | trous +0,016, objets +0,026 |

Comparaisons utiles :
- la tête dilatée seule : PixelSeal +0,006 (IC contient 0), TrustMark +0,012 ;
- 3 graines de la tête v9 : seulement +0,002 sans filtre ;
- moyenne v9 + dilatée **sans** filtre : +0,0095 et +0,014.

**Contrôles**
- StegaStamp v10 = v9 **à l'identique** (0,000 % de cellules changées), ce qui confirme que le nouvel assemblage
  reproduit l'ancien.
- Cellules changées contre v9 : PixelSeal 2,3 %, TrustMark 2,5 %.
- Fraction tatouée prédite, hors pli → test : PixelSeal 0,872 → 0,873 et 0,066 → 0,074 ; TrustMark 0,909 → 0,874 et
  0,104 → 0,097 (presque tatoué, puis partiel).

**Rejeté dans ce tour** (règle : borne basse de l'IC > 0)
- **2ᵉ graine des U-Net StegaStamp** (graine 1, mêmes réglages) : +0,002 (IC −0,008 à +0,012).
- **Recherche guidée des rectangles StegaStamp**, conçue à partir de la tolérance mesurée (la translation doit être
  juste à ~2 px, alors que la taille tolère ±8 px par bord) :
  - point de départ = boîte de la région principale du masque v9 ;
  - centre sur une grille de 4 px à ±24 px ;
  - hauteur et largeur × {0,6 ; 0,8 ; 1 ; 1,25 ; 1,6} (~4 200 boîtes, un seul passage `roi_align` + ONNX) ;
  - suppression des doublons, puis hill-climbing exact (4, 2, 1 px) des 3 meilleures ;
  - acceptation si acc ≥ 0,8.

  Résultats sur le train : **19/44 vrais rectangles** retrouvés (tous à acc ≥ 0,98, mIoU ~0,95), mais **1 fausse
  détection à acc 0,81** sur un objet transplanté (0,744 → 0,307). Gain imbriqué +0,0084 (IC **−0,0009** à +0,018),
  y compris quand le seuil est choisi en validation imbriquée → **non adopté**. Piste : un seuil fixé à l'avance à
  0,9, à valider sur de nouvelles données.

### 5.11 v11 : affinage par segments SAM des images partielles (PixelSeal, StegaStamp) [sam_masks.py, eval_v11.py, eval_v11b.py, final_v11.py]
**Pourquoi.** Les **objets transplantés** ont des contours visibles (rapport de gradient de 1,7 à 2,3), contrairement
aux trous et aux rectangles (≈ 1,0). Un segmenteur générique peut donc fournir des contours exacts, et notre carte
calibrée choisit les segments tatoués. Sur 14 objets, la meilleure combinaison de segments SAM (oracle) atteint
**0,93-0,94**, contre 0,59-0,75 pour le masque actuel.

**Segments**
- SAM ViT-B (`segment-anything` 1.0, installé à part ; poids `sam_vit_b_01ec64.pth`).
- `SamAutomaticMaskGenerator(points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.85,
  min_mask_region_area=64)`, sur l'image 512.
- Chaque segment est ramené à la grille 128 (`downsample_mask`) ; on garde ceux de 4 cellules à 0,6·N.
- Calculé pour toutes les images **classées partielles**, au train et au test (~2,2 s par image sur A100).

**Modes de sélection** (q' = carte calibrée après filtre guidé et lissage, comme en §5.10 ; masque de base = décodage
de §5.9-5.10)
- vote(τ) : union des segments de q' moyen ≥ τ.
- **glouton** (sans paramètre) : on trie les segments par q' moyen décroissant, et on n'en ajoute un à l'union que si
  le mIoU espéré ½[S/(K+T−S) + (N−K−T+S)/(N−S)] augmente (S = Σq' sur l'union, K = sa taille, T = Σq'). Union vide
  → masque de base.
- carte par segment (sans paramètre) : chaque cellule prend le q' moyen du **plus petit** segment qui la contient ;
  puis décodage du mIoU attendu et nettoyage.
- **mélange(α)** = α·q' + (1−α)·carte par segment, puis décodage.

**Validation** (imbriquée, en deux étapes : décision v10 choisie dans chaque pli sans SAM, puis mode SAM choisi
dans chaque pli parmi 9 options ; remplacement par les rectangles appliqué **en dernier, comme en test**, grâce à une
recherche de rectangles lancée sur toutes les images partielles du train) :

| méthode | mode choisi dans les 5 plis | gain | IC 95 % | objets | décision |
|---|---|---|---|---|---|
| PixelSeal | **glouton** | **+0,013** | +0,004 à +0,022 | +0,051 | ✅ |
| StegaStamp | **mélange 0,5** | **+0,004** | +0,0015 à +0,007 | +0,021 (rectangles −0,003) | ✅ |
| TrustMark | glouton ou carte par segment selon les plis | +0,006 | −0,0015 à +0,014 | +0,036 (images mal classées −0,006) | ❌ |

Le mode glouton seul est fort sur PixelSeal et TrustMark (+0,013 et +0,011), mais **nuisible** sur StegaStamp
(−0,014). Les rectangles StegaStamp ne sont pas remplacés par une recherche, et SAM les déforme.

**Contrôles**
- TrustMark v11 = v10 **à l'identique** (0,000 %).
- Cellules changées contre v10 : PixelSeal 1,3 %, StegaStamp 0,5 %.
- Fraction prédite en test, régime partiel : PixelSeal 0,068, StegaStamp 0,078.

**Rejeté dans ce tour**
- **Décodeur local du message** (`HeadLMD`) : une branche prédit les bits **par cellule de 8 px** (étages 1 à 3) ;
  son accord avec le message et sa confiance entrent dans la tête dilatée. Perte = segmentation + 0,5·BCE des bits
  (cellules tatouées) + 0,1·accord² (cellules propres). 8 epochs, 5 plis + tout le train.
  - Seul : PixelSeal −0,012 (IC −0,020 à −0,004), TrustMark −0,004.
  - Ajouté à « v9 + dilatée » : +0,0005 et +0,003 (IC contient 0).
- **Nouveaux instruments sans signal** :
  - le remover officiel de TrustMark (énergie du résidu ou corrélation au motif attendu) : AUC 0,48-0,53 ;
  - le modèle de préférence **WMForger** de Meta (carte par position à 512 et 768) : AUC 0,44-0,53.

---

## 6. Méthode d'évaluation (pourquoi croire les chiffres)
1. **Hors pli (OOF) à 5 plis stratifiés**, toujours les mêmes : aucune image n'est jugée par un modèle qui l'a vue.
2. **Epochs fixe** pour les sondes. Garder le meilleur epoch du pli de validation gonfle d'environ 0,005 ; c'est le
   cas des U-Net, à garder en tête.
3. **Même protocole** (mêmes images, même grille) pour toutes les variantes comparées.
4. **Portes de décision écrites avant l'expérience**, par exemple « +0,05 sur les trous » ou « +0,01 au global ».
5. **Biais de sélection** : choisir parmi 75 combinaisons contre 15 favorise la variante la plus riche. Avec ~170
   images par méthode, le bruit vaut ±0,005 au global et ±0,03 sur un seul pli.
6. Toujours lire le détail **par sous-type** (régime × trou, objet ou rectangle) : des gains peuvent se compenser.
7. En hors pli, la recherche de rectangle n'a tourné que sur une partie du train, alors qu'en test elle couvre toutes
   les images partielles : les estimations « avec rectangles » sont donc **pessimistes** pour ce sous-type.
8. **Règle d'adoption depuis la v9 : validation croisée imbriquée.**
   - Pour chaque pli tenu à l'écart, tous les réglages (seuils, lissage, nettoyage, poids de la moyenne, calibration)
     sont choisis sur les 4 autres plis, puis le pli est noté.
   - On calcule la différence **appariée** par image avec la référence, re-sélectionnée de la même façon, et son
     IC 95 % par bootstrap (3 000 tirages).
   - On n'adopte que si **la borne basse de l'IC est > 0**, méthode par méthode.
   - Avec ce protocole : v7 +0,086 (IC +0,068 à +0,104), v8 +0,050 (IC +0,035 à +0,066), v9 +0,026 / +0,008 / +0,009
     (IC tous > 0). La sonde StegaStamp, en revanche, fait −0,0006 (IC −0,009 à +0,008) : le +0,005 observé au départ
     n'était que du biais de sélection.

---

## 7. Procédure de reproduction (dans l'ordre)

| étape | quoi | sortie | durée (A100) |
|---|---|---|---|
| 1 | Logits globaux signés de toutes les images (§5.1) | logit signé moyen et bitacc par image | 2 min |
| 2 | Seuils de régime (§5.1) | 3 seuils (0,7988 / 0,78 / 0,755) | 1 min |
| 3 | Cartes d'accord 32→128 pour PixelSeal et TrustMark (§5.2) | une carte par image | 5 min par méthode |
| 4 | Génération synth2 : 2 graines × 3 500 par méthode (§5.3) | 7 000 images par méthode, en shards | 6 min par graine |
| 5 | Canaux auxiliaires pour réel + synthétique (§5.2) | 3 canaux par image | 10-15 min par méthode |
| 6 | U-Net : plis 0-4 + tout le train, par méthode (§5.4) | 6 modèles + OOF | 7 min par modèle |
| 7 | Choix par régime sur l'OOF (§5.5) | configuration par (méthode, régime) | 1 min |
| 8 | Liste des images test classées partielles (PixelSeal 164, TrustMark 168) | listes | — |
| 9 | Recherche de rectangle sur ces images (§5.6) | boîte + bitacc par image | 17 min (TrustMark), ~70 min (PixelSeal, à couper sur 2 GPU) |
| 10 | Assemblage v6 (§5.8) | `v6.pt` | 3 min |
| 11 | Sonde PixelSeal : 5 plis à epochs fixe + tout le train, probabilités test (§5.7) | 6 têtes, OOF, probabilités test | 1,5 min par tête |
| 12 | Intégration PixelSeal (§5.7) → v7 | `v7.pt` | 1 min |
| 13 | Sonde TrustMark, idem | 6 têtes | 1,5 min par tête |
| 14 | Intégration TrustMark avec moyenne U-Net, b ∈ {1 ; 0,75 ; 0,5} → v8 | **`v8.pt`** | 3 min |
| 15 | TTA : pour chacun des 6 modèles de chaque famille (U-Net StegaStamp, U-Net TrustMark, sondes PixelSeal, sondes TrustMark), probabilités sans et avec TTA sur son pli de validation et sur le test (§5.9) | cartes par modèle | ~1 min par famille |
| 16 | Évaluation imbriquée (TTA, décodage) et contrôle « sans TTA = OOF stocké » (§5.9, §6) | gains + IC | 3 min (CPU) |
| 17 | Assemblage v9 : calibration par famille et régime sur toutes les cartes hors pli avec TTA ; en test, calibration par modèle, puis moyenne, décodage et rectangles (§5.9) | **`v9.pt`** | 2 min (CPU) |
| 18 | Têtes dilatées (8 epochs, 5 plis + tout le train) pour PixelSeal et TrustMark, puis leur TTA (§5.10) | 6 têtes par méthode | 2,4 min par tête |
| 19 | Évaluation imbriquée des variantes de cartes et du filtre guidé, variante choisie dans la validation imbriquée (§5.10) | choix par méthode | 10 min (CPU) |
| 20 | Assemblage v10 : unités « v9 + dilatée », calibration, filtre guidé, décodage, rectangles ; StegaStamp identique à la v9 (§5.10) | **`v10.pt`** | 5 min (CPU) |
| 21 | Segments SAM (« everything », 32 points par côté) des images classées partielles, au train et au test (§5.11) | segments par image | ~2,2 s par image |
| 22 | Recherche de rectangles sur toutes les images partielles du train (PixelSeal, TrustMark), pour évaluer SAM avec le remplacement par les rectangles | boîtes du train | 8 min (TrustMark), 12 min (PixelSeal, sur 3 GPU) |
| 23 | Évaluation imbriquée en 2 étapes (décision v10, puis mode SAM) avec rectangles (§5.11) | choix par méthode | 10 min |
| 24 | Assemblage v11 (§5.11) | **`v11.pt`** | 5 min |
| 25 | Libérer le GPU | — | — |

Sur 4 GPU, paralléliser les tâches indépendantes (méthodes, plis, graines). Les étapes 11 à 14 seules prennent
~15 min ; les étapes 15 à 17, ~10 min.

**Chiffres de contrôle** (un écart net signale une différence d'implémentation) :

| contrôle | attendu |
|---|---|
| v1 (régime seul) sur le train | 0,4526 |
| U-Net de pli, validation @0,5 | PixelSeal ~0,46-0,47 ; TrustMark 0,50-0,53 ; StegaStamp 0,655-0,704 |
| OOF des U-Net, choix du §5.5 | StegaStamp partiel 0,635 / presque tatoué 0,744 ; TrustMark 0,513 / 0,535 |
| Rectangles acceptés (bitacc ≥ 0,8) | TrustMark 79/168 ; PixelSeal 38/164 |
| Sonde PixelSeal, plis à epochs fixe (@0,5, sans rectangles) | 0,579 / 0,568 / 0,561 / 0,561 / 0,590 |
| Intégration PixelSeal (OOF) | presque tatoué : t 0,7, c 16 → 0,600 ; partiel : t 0,3, c 16 → 0,546 ; global 0,501 → **0,587** |
| Sonde TrustMark, plis à epochs fixe | 0,584 / 0,540 / 0,574 / 0,599 / 0,538 |
| Intégration TrustMark (OOF) | presque tatoué : t 0,7, c 0, b 0,75 → 0,595 ; partiel : t 0,3, c 16, b 0,75 → 0,559 ; global 0,565 → **0,619** |
| v8 contre v6 | 264/340 masques PixelSeal et 240/340 TrustMark modifiés ; StegaStamp identique |
| contrôle TTA | |probabilité sans TTA − OOF stocké| ≤ 0,0004 en moyenne |
| v9, validation imbriquée (sans rectangles) | StegaStamp 0,690 → 0,715 ; TrustMark 0,577 → 0,585 ; PixelSeal 0,574 → 0,583 |
| v9, réglages (s, c) | partout (1, 16), sauf StegaStamp partiel (0, 0) |
| v9, fraction tatouée prédite en test | PixelSeal 0,878 / 0,067 ; TrustMark 0,883 / 0,092 ; StegaStamp 0,873 / 0,077 (presque tatoué / partiel) |
| v9 contre v8 | cellules modifiées : 2,8 % (PixelSeal), 2,3 % (TrustMark), 1,7 % (StegaStamp) |
| v10, validation imbriquée | PixelSeal 0,583 → 0,595 ; TrustMark 0,585 → 0,603 ; StegaStamp inchangé (0,715) |
| v10, réglages (r, ε ; s ; c) | PixelSeal (4 ; 0,01 ; 1 ; 16) / (8 ; 0,001 ; 1 ; 16) ; TrustMark (8 ; 0,01 ; 1 ; 16) / (8 ; 0,001 ; 0 ; 0) (presque tatoué / partiel) |
| v10 contre v9 | cellules modifiées : 2,3 % (PixelSeal), 2,5 % (TrustMark), 0,000 % (StegaStamp) |
| v11, validation imbriquée avec rectangles | PixelSeal 0,632 → 0,645 (glouton) ; StegaStamp 0,715 → 0,719 (mélange 0,5) ; TrustMark inchangé |
| v11 contre v10 | cellules modifiées : 1,3 % (PixelSeal), 0,000 % (TrustMark), 0,5 % (StegaStamp) |

---

## 8. Résultats et historique

| version | contenu | estimation |
|---|---|---|
| tout à zéro | — | 0,25 |
| v1 | régime seul | 0,45 |
| v2-v3 | U-Net sur synthétique naïf, puis canaux auxiliaires | ~0,55-0,56 |
| v4-v5 | + recherche de rectangle, + générateur synth2 | ~0,62 |
| v6 | ensembles de 6 U-Net + choix hors pli par régime | ~0,62 |
| v7 | + sonde PixelSeal | ~0,64 |
| v8 | + sonde TrustMark (b = 0,75) | ~0,66 |
| v9 | + TTA (4 translations) + calibration isotone par modèle + décodage du mIoU attendu | ~0,68 |
| v10 | + tête dilatée moyennée avec la tête v9, + filtre guidé (PixelSeal, TrustMark) | ~0,69 |
| **v11** | + segments SAM en régime partiel : glouton (PixelSeal), mélange 0,5 (StegaStamp) | **~0,695** |

| sous-type (hors pli) | trous (49 %) | objets partiels (25 %) | rectangles partiels (25 %) |
|---|---|---|---|
| PixelSeal v6 → v8 | 0,476 → **0,593** | 0,527 → **0,629** | ~0,66 avec rectangles (~44 % décodables) |
| TrustMark v6 → v8 | 0,531 → **0,590** | 0,502 → **0,609** | ~0,93 sur les rectangles trouvés (~100 %) |
| StegaStamp (U-Net) | 0,744 | 0,645 | 0,627 |
| **v9 contre v8** (validation imbriquée, sans rectangles) | StegaStamp +0,011, TrustMark +0,011, PixelSeal +0,010 | StegaStamp +0,024, TrustMark +0,005, PixelSeal +0,015 | StegaStamp **+0,055** (sous-segmentation corrigée), TrustMark +0,003, PixelSeal −0,001 |

---

## 9. Pourquoi la sonde marche (à retenir)
- Un U-Net générique sur les pixels ne « voit » pas un tatouage **imperceptible** (PixelSeal : 0,46).
- Le décodeur officiel, lui, a été entraîné à le détecter, mais sa sortie est trop grossière (8×8, champ récepteur
  quasi global).
- Ses couches intermédiaires combinent les deux qualités : **sensibles au motif** et **localisées** (8 à 32 px).
- Pour un tatouage **fort** (StegaStamp), le U-Net suffit et la sonde n'apporte rien (§10).

---

## 10. Ce qui n'a PAS marché (avec les paramètres essayés)

| tentative | résultat | raison |
|---|---|---|
| Générateur v1 : objets déplacés, ou recadrage tatoué à son échelle pour les objets | carte nulle sur les objets synthétiques | contraire à F4 |
| U-Net sur RGB pour PixelSeal | 0,46-0,47 | motif imperceptible pour un encodeur ImageNet |
| Filtre adapté par re-tatouage (PixelSeal, TrustMark) : résidu flottant, luminance, passe-bande DoG σ 1-6 px, fenêtres de 16/32 px, normalisation par messages nuls, différence « message − complément » | AUC 0,49-0,54, **même sur synthétique** | résidu anti-corrélé au contenu (F8) |
| Gradient × entrée (normalisé par 4 messages nuls) | AUC ~0,49 | — |
| Canal de détection 0 par position (PixelSeal) | AUC 0,53 (trous), 0,46 (objets) | — |
| Occlusion d'une zone à la fois (zones de 64/48/32 px, flou σ = 6, la valeur nécessaire pour effacer PixelSeal) | AUC 0,73-0,74 contre 0,646, mais mIoU borne haute +0,02 seulement | trop flou pour des trous de ~10 % |
| Masques aléatoires (RISE, cellules de 32 px, régression ridge) | AUC 0,53 (50 % gardé), 0,69 (85 % gardé, bords doux) | détecteur hors de son régime linéaire, artefacts aux bords |
| Affinage des U-Net sur vraies images seules | aucun gain | — |
| Recherche de rectangle StegaStamp (grille globale de pas côté/2 ; départ depuis la boîte du U-Net) | aucun vrai succès (0,6-0,7 = bruit du maximum) | F6 |
| **Sonde StegaStamp** : graphe ONNX avec sorties `dec_conv_4/5/6/7` (64@100², 64@50², 128@25², 128@13²) via ONNX Runtime (IO binding) ; accord 13×13 par linéarisation exacte des 2 couches denses | seule 0,55 contre U-Net 0,69 ; meilleure moyenne +0,005, mais **−0,0006 en validation imbriquée** (IC −0,009 à +0,008) → **rejetée** | AUC de la sonde très inférieure au U-Net (rectangles 0,585 contre 0,906 ; trous 0,865 contre 0,951) ; complémentarité faible |
| Décodage par « surface attendue » (garder les round(Σq) meilleures cellules) | StegaStamp +0,006 (IC contient 0), TrustMark 0, PixelSeal +0,001 | inférieur au décodage du mIoU attendu |
| Rectangularisation (remplacer une région partielle rectangulaire par la boîte de la région q ≥ q_bas ; τ ∈ {0,7 ; 0,8 ; 0,9}, q_bas ∈ {0,15 ; 0,3}) | aucun gain au-delà du décodage | — |
| Recherche locale fine des rectangles StegaStamp (centre au pas de 2 px) | **non lancée** : sa condition d'entrée échoue | même après le nouveau décodage, seuls 25 % des rectangles décodables ont un point de départ à ±12 px du centre (erreur médiane 44 px). Pourtant le décodeur tolère qu'on déplace **un** bord (4 px → acc 0,84 ; 8 px → 0,75) ; seule la **translation** est critique (4 px → 0,60) |
| 2ᵉ graine des U-Net StegaStamp (moyenne de 2 graines) | +0,002 (IC −0,008 à +0,012) | les U-Net StegaStamp sont déjà stables |
| 3 graines de la tête de sonde v9 (sans tête dilatée) | +0,002 sans filtre ; dominé par « v9 + dilatée » | peu de diversité (même synthétique) |
| Recherche **guidée** des rectangles StegaStamp (centre au pas de 4 px à ±24 px × tailles ×{0,6…1,6}, puis affinage) | 19/44 rectangles (acc ≥ 0,98), mais 1 fausse détection à 0,81 → +0,0084 (IC −0,0009 à +0,018) → **rejetée** | une fausse détection coûte ~0,44 sur son image ; voir les pistes (§12) |
| Décodeur local du message (bits prédits par cellule de 8 px, accord avec le message en entrée de la tête dilatée) | seul : PixelSeal −0,012, TrustMark −0,004 ; avec « v9 + dilatée » : +0,0005 et +0,003 (IC contient 0) | l'indice fin lié au message n'apporte rien de plus que les couches gelées |
| Remover officiel TrustMark (énergie du résidu ; corrélation au motif attendu) | AUC 0,48-0,53 | modifie l'image partout de la même façon |
| Modèle de préférence WMForger (Meta, NeurIPS 2025), carte par position à 512 et 768 | AUC 0,44-0,53 | détecteur générique sans signal localisé sur ces 3 tatouages |
| SAM pour TrustMark | +0,006 (IC −0,0015 à +0,014) → rejeté | instabilité du mode choisi selon les plis ; images mal classées |

---

## 11. Réappliquer la méthode (démarche générale)
1. **Lister ce qui est connu en test** et l'exploiter jusqu'au bout (ici : méthode et message).
2. **Bilan des pertes par sous-type** (régime × forme), avec bornes hautes (oracle) et références (constantes). On
   attaque le plus gros poste.
3. **Reconstituer le processus de génération** avec les décodeurs comme instruments :
   - formuler des variantes ;
   - en générer un petit lot chacune ;
   - comparer leurs statistiques au réel ;
   - corriger le générateur.
4. **Tatouage imperceptible** → sonde sur le décodeur officiel gelé. **Tatouage fort** → U-Net classique.
5. **Hypothèse vérifiable par décodage** (rectangle à son échelle) → recherche explicite (énumération massive, puis
   affinage), avec un seuil d'acceptation sans fausse détection en validation.
6. **Évaluation honnête** :
   - hors pli, epochs fixe, même protocole pour tous ;
   - portes de décision écrites avant ;
   - attention au biais de sélection ;
   - **un contrôle synthétique** pour distinguer « méthode cassée » et « données différentes ».
7. **Avant chaque gros chantier** : un plan écrit, challengé 3 fois (signal réel ? faisable ? fait bouger la
   métrique ?).
8. **Infrastructure** :
   - un seul job d'équipe, sorties dans le home ;
   - reprise après crash, données en shards ;
   - libérer le nœud vite.

## 12. Pistes non tentées
- PixelSeal : les petits ou fins rectangles non décodables (~14 % des images).
- StegaStamp : les rectangles à leur échelle, par une recherche sous-pixel ou en exploitant le module de recalage
  spatial.
  Il faudrait d'abord de bien meilleurs points de départ (centre à ±12 px).
- Rectangles StegaStamp : la recherche guidée (§5.10) en retrouve 19 sur 44, avec une seule fausse détection à acc 0,81.
  Un seuil fixé à l'avance à 0,9, puis validé sur des données neuves, la rendrait probablement rentable (≈ +0,012 sur
  StegaStamp).
- TTA avec plus de translations ou de petits zooms ; plus de diversité synthétique pour les sondes.
- La marge de seuil restante (borne haute par image) après la v9 reste de l'ordre de +0,02 à +0,03 par méthode.
  Un réseau de raffinement (carte de probabilité + image → masque), entraîné hors pli, pourrait en capturer une
  partie.
- SAM pour TrustMark : +0,006 (IC touchant 0). Fixer à l'avance un mode unique (glouton : +0,011 seul) et limiter
  SAM aux images dont le régime partiel est sûr (bit accuracy très basse) pourrait le faire passer.
- Confirmer les estimations (~0,68 pour la v9, ~0,69 pour la v10, ~0,695 pour la v11) sur le leaderboard public
  (30 % du test).
