# Unlearning : solutions existantes et littérature (recherche du 22/11)

Le tri suit nos contraintes : **forget set seul (pas de retain), il faut imiter gold (pas seulement chuter en accuracy), il faut préserver le control**.
Pour chaque entrée, « → nous » dit ce qu'on en tire concrètement. Les détails marqués *(lu)* viennent de la lecture du papier.
Les autres viennent des abstracts et des pages de résultats.

## A. Unlearning sans retain set (notre cas)

1. **SCAR, « Is Retain Set All You Need in Machine Unlearning? »** — Bonato, Cotogni, Sabetta, ECCV 2024.
   [arXiv 2404.12922](https://arxiv.org/abs/2404.12922), code [github.com/jbonato1/SCAR](https://github.com/jbonato1/SCAR). *(lu)*
   - Perte `λ1·L_M + λ2·L_TD`.
   - `L_M` pousse l'embedding de chaque image forget vers la distribution (Mahalanobis) de la **classe étrangère la plus proche**.
   - `L_TD` distille l'original en divergence de Jensen-Shannon sur des **images hors distribution** (sous-ensemble d'ImageNet), qui remplacent le retain set.
   - Réglages CIFAR : λ1=1, **λ2=5**, lr 5e-4, T=1, au plus 30 epochs.
   - Gère le retrait de classe et le retrait aléatoire de 10 % (« HR »).
   - → nous : c'est le schéma de nos ancres Imagenette, mais avec la distillation pesée 5× plus que l'oubli. Cible « classe voisine » pour les 9 classes retirées.
2. **JiT, « Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization »**, renommé « An Information Theoretic Approach to Machine Unlearning » — Foster, Fogarty, Schoepf, Öztireli, Brintrup, 2024.
   [arXiv 2402.01401](https://arxiv.org/abs/2402.01401), code [github.com/jwf40/Information-Theoretic-Unlearning](https://github.com/jwf40/Information-Theoretic-Unlearning). *(lu)*
   - N'utilise **que le forget**.
   - Minimise `mean_j ||f(x) − f(x+ξ_j)||₂ / ||ξ_j||₂` avec `ξ ~ N(0, σ²)`, une epoch.
   - ViT, oubli aléatoire : η=0.01, σ=0.5. Le retain bouge d'environ 1 %, avec un léger sur-oubli.
   - Testé sur ViT, VGG11 et ViT-L/ImageNet.
   - → nous : baseline peu coûteuse (une soumission). Lisser la sortie autour de x retire le « pic » de mémorisation, et aide aussi sur les copies altérées.
3. **Learning to Unlearn: Instance-wise Unlearning for Pre-trained Classifiers** — Cha et al., AAAI 2024. [arXiv 2301.11578](https://arxiv.org/abs/2301.11578)
   - Unlearning **par instance avec le forget seul**.
   - Des exemples adversariaux générés depuis les images forget (PGD L2 ciblé, 100 itérations, ε=0.4), étiquetés par l'original, servent de proxy de retain.
   - S'y ajoute une régularisation par importance des poids.
   - → nous : ancres « près de la frontière », gratuites à générer.
4. **ZS-PAG, Zero-Shot Machine Unlearning with Proxy Adversarial Data Generation** — Chen et al., IJCAI 2025. [arXiv 2507.21738](https://arxiv.org/abs/2507.21738) *(lu)*
   - Proxys adversariaux (PGD vers la **2e classe** de l'original).
   - Sous-espace par couche (SVD des activations des proxys), avec projection du gradient `P = I − SSᵀ` (à la manière de GPM).
   - Pseudo-labels guidés par l'influence.
   - Testé seulement en retrait de classe.
   - → nous : projeter les mises à jour hors du sous-espace des ancres limite mécaniquement la dérive du control.
5. **Zero-Shot Machine Unlearning** (référence des organisateurs) — Chundawat, Tarun, Mandal, Kankanhalli, IEEE TIFS 2023.
   [arXiv 2201.05629](https://arxiv.org/abs/2201.05629), code [github.com/ayu987/zero-shot-unlearning](https://github.com/ayu987/zero-shot-unlearning)
   - Bruit qui minimise ou maximise l'erreur, et « Gated Knowledge Transfer ».
   - Pensé pour le retrait de classe, faible en retrait aléatoire selon Foster et al.
6. **Boundary Unlearning** — Chen et al., CVPR 2023. [arXiv 2303.11570](https://arxiv.org/abs/2303.11570)
   - *Boundary Shrink* : chaque image forget est réétiquetée vers sa **classe incorrecte la plus proche**, trouvée avec un bruit FGSM puis la prédiction de l'original. Forget seul, retrait de classe.
   - → nous : cible des 9 classes retirées (« quelle mauvaise classe »).
7. **Zero-shot class unlearning in CLIP with synthetic samples** — Kravets & Namboodiri, 2024. [arXiv 2407.07485](https://arxiv.org/abs/2407.07485)
   - Génère des échantillons synthétiques par montée de gradient (inversion de modèle). Utile seulement si on manquait d'images.

## B. Imiter gold : savoir quelles images le modèle ré-entraîné raterait

8. **Twin Machine Unlearning (TMU)** — Sun et al. [arXiv 2408.11433](https://arxiv.org/abs/2408.11433) / [2501.08615](https://arxiv.org/abs/2501.08615) *(lu)*
   - Construit un problème « jumeau » où le gold est connu, puis apprend un prédicteur « facile ou difficile » par image (gold la classera-t-il bien ?).
   - Il le transfère ensuite au vrai forget set, et maximise la perte des images difficiles en préservant les faciles.
   - Les features utilisées : distance kNN aux features du retain, **robustesse adversariale** (CE entre logits propres et adversariaux), perte d'un modèle brièvement entraîné.
   - Résultat : écart d'accuracy au gold de 4.2 pts, contre 8 à 13 pour les baselines (CIFAR-10). En ViT, 3.3.
   - → nous : notre « difficulté par image » doit s'inspirer de ces features. La robustesse adversariale et le kNN se calculent avec le seul modèle orig.
9. **C-score, « Characterizing Structural Regularities of Labeled Data »** — Jiang, Zhang, Talwar, Mozer, ICML 2021.
   [arXiv 2002.03206](https://arxiv.org/abs/2002.03206), [site](https://pluskid.github.io/structural-regularity)
   - Le C-score est l'accuracy *attendue* d'un modèle qui n'a pas vu l'image : exactement la quantité gold par image.
   - Proxys à un seul modèle : densité et distance par paires dans l'espace caché. La vitesse d'apprentissage, elle, demande l'historique d'entraînement.
10. **Prediction depth** — Baldock, Maennel, Neyshabur, NeurIPS 2021. [arXiv 2106.09647](https://arxiv.org/abs/2106.09647)
    - Des sondes kNN à chaque couche montrent que les images faciles sont décidées tôt, et les difficiles ou mémorisées tard.
    - → nous : la difficulté se mesure par des sondes kNN couche par couche sur le forget lui-même.
11. **Memorization Through the Lens of Curvature of Loss Function Around Samples** — Garg, Ravikumar, Roy, ICML 2024. [arXiv 2307.05831](https://arxiv.org/abs/2307.05831)
    - Une forte courbure de la perte par rapport à l'**entrée** signale une image atypique ou mal étiquetée, donc mémorisée.
12. **What Neural Networks Memorize and Why** — Feldman & Zhang, NeurIPS 2020. [arXiv 2008.03703](https://arxiv.org/abs/2008.03703)
    - Mémorisation = accuracy avec moins accuracy sans l'image. Les exemples atypiques sont ceux que le modèle ré-entraîné rate.
13. **Inexact Unlearning Needs More Careful Evaluations (U-LiRA)** — Hayes, Shumailov, Triantafillou, Khalifa, Papernot, 2024. [arXiv 2403.01218](https://arxiv.org/abs/2403.01218)
    - L'évaluation par exemple est bien plus sévère que l'évaluation en population. C'est pourquoi « matching only the average loss is not enough ».
14. **On the limitation of evaluating machine unlearning using only a single training seed** — Lanyon et al., 2025. [arXiv 2510.26714](https://arxiv.org/abs/2510.26714)
    - La variance entre seeds d'entraînement est forte. Ici, gold est une moyenne de plusieurs modèles, donc une distribution lisse : préférer des cibles moyennées.

## C. Localiser la modification pour préserver le control

15. **Localizing Memorization in SSL Vision Encoders** (groupe des superviseurs, CISPA) — Wang, …, Dziedzic, Boenisch, NeurIPS 2024. [arXiv 2409.19069](https://arxiv.org/abs/2409.19069)
    - Métriques LayerMem et UnitMem. Dans les ViT, la mémorisation est surtout dans les **couches fully-connected (MLP)** et croît avec la profondeur. Les points atypiques sont plus mémorisés.
    - → nous : n'entraîner que les MLP des derniers blocs, ou du LoRA dessus.
16. **Can Neural Network Memorization Be Localized?** — Maini et al., ICML 2023.
    [arXiv 2307.09542](https://arxiv.org/abs/2307.09542), code [github.com/pratyushmaini/localizing-memorization](https://github.com/pratyushmaini/localizing-memorization)
    - La mémorisation tient dans peu de neurones répartis dans le réseau. Les couper fait chuter les exemples mémorisés de 100 % à 3 %, avec peu d'effet sur les autres.
17. **SalUn** (référence des organisateurs) — Fan et al., ICLR 2024.
    [arXiv 2310.12508](https://arxiv.org/abs/2310.12508), code [github.com/OPTML-Group/Unlearn-Saliency](https://github.com/OPTML-Group/Unlearn-Saliency)
    - Masque de saillance des poids (gradient de la perte forget) : on ne met à jour que les k % de poids les plus saillants.
18. **SSD / LFSSD, Selective Synaptic Dampening** — Foster, Schoepf, Brintrup, AAAI 2024.
    [arXiv 2308.07707](https://arxiv.org/abs/2308.07707), code [github.com/if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening)
    - Atténue les poids dont l'importance de Fisher est disproportionnée pour le forget, sans ré-entraînement, en quelques minutes.
    - → nous : calculer l'importance « retain » sur les ancres.
19. **SCRUB, « Towards Unbounded Machine Unlearning »** (référence des organisateurs) — Kurmanji et al., NeurIPS 2023.
    [arXiv 2302.09880](https://arxiv.org/abs/2302.09880), code [github.com/meghdadk/SCRUB](https://github.com/meghdadk/SCRUB)
    - Professeur-élève max-min : on s'éloigne sur le forget et on reste proche sur le retain. Utilisable avec les ancres comme retain de substitution.
20. **NPO** (référence des organisateurs) — Zhang, Lin, Bai, Mei, COLM 2024.
    [arXiv 2404.05868](https://arxiv.org/abs/2404.05868), code [github.com/licong-lin/negative-preference-optimization](https://github.com/licong-lin/negative-preference-optimization)
    - Montée de gradient bornée. Variante sans référence : SimNPO, [github.com/OPTML-Group/Unlearn-Simple](https://github.com/OPTML-Group/Unlearn-Simple).
    - Version classification décrite dans IMU, [arXiv 2508.01620](https://arxiv.org/abs/2508.01620).
    - → nous : si on ajoute un terme « repousser », le faire en NPO plutôt qu'en montée de gradient brute.
21. **NatMU, Towards Natural Machine Unlearning** — He et al., 2024. [arXiv 2405.15495](https://arxiv.org/abs/2405.15495)
    - Injecte par mixup des features « correctes » dans les images forget pour éviter le sur-oubli.
22. **Task arithmetic** — Ilharco et al., ICLR 2023. [arXiv 2212.04089](https://arxiv.org/abs/2212.04089)
    - Soustraire un « task vector » (orig fine-tuné sur F, moins orig). Baseline de 10 min.

## D. Compétitions et benchmarks proches

23. **NeurIPS 2023 Machine Unlearning Challenge**, « Are we making progress in unlearning? » — Triantafillou et al., 2024. [arXiv 2406.09073](https://arxiv.org/abs/2406.09073) *(lu)*
    - Métrique : confiance de la bonne classe *par exemple*, comparée à 512 modèles ré-entraînés.
    - Méthodes du haut du classement :
      - Fanchuan : KL vers l'uniforme sur le forget, puis montée contrastive forget/retain, puis CE sur le retain.
      - Kookmin : ré-initialisation des poids conv à faible gradient, puis réparation sur le retain.
      - Seif : bruit gaussien σ=0.6 sur les poids conv, puis fine-tuning sur le retain.
      - Sebastian : ré-initialisation de 99 % des poids de plus petite norme, CE sur le retain, et appariement de l'entropie à l'original.
      - Amnesiacs : ré-initialisation de la 1re conv et de la tête, distillation de l'original sur des données tenues à l'écart, CE et KL symétrique.
    - **Toutes utilisent le retain set**, que nous n'avons pas. L'enseignement est que la « réparation » doit chez nous être une distillation vers orig sur des données de substitution.
24. **Évaluation sous transformations** (cf. nos « hidden altered copies ») : [arXiv 2411.15477](https://arxiv.org/abs/2411.15477) et [arXiv 2510.09007](https://arxiv.org/abs/2510.09007).
    - Un unlearning évalué seulement sur les entrées exactes surestime l'oubli. Il faut entraîner et évaluer sur plusieurs familles d'altérations.

## E. Données de substitution

- **Toward Efficient Data-Free Unlearning** [arXiv 2412.13790](https://arxiv.org/abs/2412.13790) : générateur de distillation sans données, en filtrant les échantillons de la classe oubliée.
- Pour nous, plus simple : un générateur texte→image (SD-Turbo) avec des prompts au style du dataset (piste D du plan).

## Ce que je retiens, par ordre de priorité

1. **SCAR et les superviseurs (LayerMem)** : distillation lourde vers orig sur des ancres, et mise à jour limitée aux MLP des derniers blocs. Cela traite la cause n°1 du blocage (retain).
2. **Structure des classes** : trancher H0/H1 avec les sondes. Pour les classes retirées, chirurgie analytique ou classe voisine (Boundary Shrink / SCAR).
3. **TMU, C-score et prediction depth** : estimer *quelles* images gold rate (robustesse adversariale, kNN multi-couches), pour le score de prédiction.
4. **JiT** (forget seul) et **SSD** : baselines peu coûteuses, une soumission chacune au plus, si un GPU est libre.
