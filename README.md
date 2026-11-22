# TP2 — Modèles génératifs avec PyTorch (MNIST)

**Cours :** Modèles Génératifs — Par Bounjoume Oussama
**Outil :** PyTorch + UV (gestion d'environnement) + Jupyter / VS Code

## Objectif

Créer, entraîner, tester et explorer les modèles génératifs sur le dataset **MNIST** :

1. Charger et explorer le dataset MNIST
2. Créer, entraîner et évaluer un **modèle discriminatif** $P(y|x)$
3. Créer, entraîner, évaluer et explorer un **autoencodeur simple (AE)**
4. Créer, entraîner, évaluer et explorer un **autoencodeur variationnel (VAE)**

Le notebook complet se trouve dans [`notebooks/modeles_generatifs_mnist.ipynb`](notebooks/modeles_generatifs_mnist.ipynb).

## Configuration de l'environnement (UV)

```bash
# 1. Créer le projet et l'environnement virtuel
uv init --python=3.11
uv venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\Activate       # Windows

# 2. Ajouter les dépendances
uv add ipykernel
uv add torch torchvision torcheval matplotlib numpy scikit-learn

# 3. Lancer le notebook
jupyter notebook notebooks/modeles_generatifs_mnist.ipynb
```

## Structure du notebook

| Partie | Contenu | Modèle | Loss |
| -------- | --------- | -------- | ------ |
| **I** | Chargement MNIST (60 000 train / 10 000 test, 28×28) | — | `ToTensor()` normalisation [0,1] |
| **II** | **Modèle discriminatif** : MLP 784 → 128 → 10 | `Classifier` | `CrossEntropyLoss` |
| **III** | **Autoencodeur** : encodeur 784→128→32→2, décodeur miroir avec sigmoïde | `Autoencoder` (latent d=2) | `MSELoss` |
| **IV** | **VAE** : encodeur produit (μ, σ), reparameterization trick, décodeur | `VAE` | MSE (reconstruction) + KL (régularisation) |
| **V** | Synthèse comparative, limites du AE vs VAE, exercices | — | — |

## Résultats obtenus

### 1. Modèle discriminatif (Partie II)

- Architecture : `Flatten → Linear(784,128) → ReLU → Linear(128,10)`
- Optimiseur : Adam (lr = 1e-3), 5 époques, batch 128
- **Accuracy test ≈ 97.25 %**
- Analyse : matrice de confusion (`sklearn`) + rapport de classification + visualisation des erreurs (chiffres ambigus : 4/9, 3/8, 7/1)
- **Limite fondamentale** : le modèle apprend $P(y|x)$, il discrimine mais **ne peut pas générer** d'images.

### 2. Autoencodeur (Partie III)

- Objectif : $x \to z \to \hat{x}$ avec $\hat{x} \approx x$, latent volontairement réduit à **d = 2** pour visualiser l'espace latent en 2D
- MSE de reconstruction ≈ **0.039** après 20 époques
- Visualisation : reconstructions (originales vs reconstruites) et projection 2D de l'espace latent colorée par classe
- **Limites observées** : l'espace latent n'est **pas régularisé** — les points sont groupés par classe mais les zones entre classes n'ont aucune garantie de produire un chiffre valide. Échantillonner un $z$ aléatoire donne une reconstruction médiocre.

### 3. VAE (Partie IV)

- L'encodeur produit une **distribution** $\mathcal{N}(\mu, \sigma^2)$ au lieu d'un point fixe
- **Reparameterization trick** : $z = \mu + \sigma \odot \varepsilon$ avec $\varepsilon \sim \mathcal{N}(0, I)$ (permet la rétropropagation)
- Loss complète : $\mathcal{L} = \mathcal{L}_{rec} + \beta \cdot D_{KL}\big(q_\phi(z|x) \,\|\, p(z)\big)$
- Effet de la régularisation KL : l'espace latent devient **continu et structuré** → on peut échantillonner $z \sim \mathcal{N}(0,I)$ et générer de **nouveaux** chiffres plausibles, et interpoler entre deux chiffres
- **Compromis clé** : reconstruction fine (AE) ↔ espace latent exploitable pour la génération (VAE)

## Synthèse : discriminatif vs génératif

| | Discriminatif | Autoencodeur | VAE |
| --- | --- | --- | --- |
| Apprend | $P(y\|x)$ | $x \to \hat{x}$ | $p(x)$ via $q(z\|x)$ |
| Labels requis | Oui | Non | Non |
| Espace latent | — | non régularisé | régularisé (gaussien) |
| Génération de nouvelles images | Non | limité | Oui |

## Ce qu'il faut retenir

- On passe de **Reconnaître** (classifieur) → **Représenter** (AE) → **Générer** (VAE).
- Le goulot d'étranglement ($d \ll 784$) force le modèle à apprendre une représentation compacte.
- La divergence KL est le prix à payer pour rendre l'espace latent **probabiliste et échantillonnable**.

## Pistes d'approfondissement

- GAN (générateur/discriminateur antagonistes) et modèles de diffusion (bruitage/débruitage)
- Latent de dimension plus grande (16/32) pour de meilleures reconstructions
- Conditional VAE (génération conditionnée par le label)
- Entraînement plus long / scheduling du learning rate

## Références

- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- Documentation PyTorch : <https://pytorch.org/docs/>
