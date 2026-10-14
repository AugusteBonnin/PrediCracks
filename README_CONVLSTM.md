# Prédiction temporelle des craquelures avec ConvLSTM

Le modèle reçoit une fenêtre de 5 images consécutives avec leur temps normalisé et
prédit l'image immédiatement suivante. L'inférence réinjecte chaque prédiction dans
la fenêtre, ce qui produit une évolution autorégressive.

Les fichiers binaires `.data` du jeu de données sont rangés dans `data/`. Les
scripts utilisent ce dossier par défaut ; l'option `--data-dir` permet d'en choisir
un autre.

## Installation et entraînement

```bash
cd ~/Documents/279
# Activer ici l'environnement qui contient déjà Keras et PlaidML.
plaidml-setup  # nécessaire une seule fois pour sélectionner le périphérique
python train_convlstm.py
```

Le meilleur modèle est écrit dans `best_convlstm_delta.h5`. Le journal des époques
et les métadonnées utilisent le même préfixe. Ce modèle apprend explicitement les
variations sur des horizons de 1 à 5 images, avec une pondération renforcée des
pixels qui évoluent.

Si la mémoire GPU est insuffisante :

```bash
python train_convlstm.py --batch-size 1
```

## Prédiction progressive

```bash
python inference_convlstm.py --model best_convlstm_delta.h5 --steps 100
```

Les cinq premières images réelles servent par défaut à initialiser l'état temporel.
Les sorties PNG visualisables, les tableaux NumPy non clippés et les temps sont
placés dans `convlstm_predictions/`. Utiliser `--start-index N` pour choisir une
autre fenêtre initiale.

Pour initialiser la séquence avec un bruit binaire contenant exactement 12,5 % de
pixels noirs :

```bash
python inference_convlstm.py --noise-black-fraction 0.125 --steps 100
```

La graine est enregistrée dans `seed.png` et `seed.npy`. L'option `--noise-seed`
permet de produire un autre tirage reproductible.

Le découpage entraînement/validation est chronologique : les 20 % dernières images
ne servent jamais de cibles pendant l'entraînement.
