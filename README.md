# 🌿 Plant Disease Detection Using Machine Learning

Ce projet vise à résoudre le problème de la détection de maladies des plantes à l’aide de techniques de Machine Learning.  
L’objectif est de construire un modèle capable de **classifier automatiquement les feuilles de pommier** en deux catégories : **saines** ou **malades**.

> ⚠️ Ce projet a été conçu à des fins d’apprentissage. Il ne prétend pas être la solution la plus avancée au problème, mais constitue une excellente base pour explorer l’application de l’IA en agriculture.

---

## 📂 Dataset Overview

Le jeu de données provient du [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color) et contient des **images couleur de feuilles de pommier** :

- 📁 **Healthy** : Feuilles vertes, en bonne santé
- 📁 **Diseased** : Feuilles atteintes de maladies telles que :
  - Apple Scab
  - Black Rot
  - Cedar Apple Rust

---

## 🖼️ Propriétés des Images

- 📷 **Format** : JPG
- 📏 **Taille** : 256x256 pixels
- 🔍 **Résolution** : 96 dpi
- 🎨 **Profondeur des couleurs** : 24 bits

---

## 🧪 Méthodologie

### 🧹 Prétraitement

- Chargement des images
- Conversion BGR → RGB → HSV pour une meilleure séparation couleur/intensité
- **Segmentation** pour extraire les couleurs spécifiques des feuilles

### 🧬 Extraction de caractéristiques

- **Couleur** : Histogrammes de couleurs (moyennes, variances)
- **Forme** : Moments de Hu
- **Texture** : Haralick (co-occurrence de niveaux de gris)

Les caractéristiques sont ensuite concaténées à l’aide de `NumPy` et normalisées via **MinMaxScaler**.

### 🧠 Entraînement des modèles

- Découpage du dataset :
  - **80%** pour l'entraînement
  - **20%** pour les tests
- Sauvegarde des caractéristiques dans un fichier `.hdf5`
- **Validation croisée (k=10)** pour chaque modèle

### 🤖 Modèles testés

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier ✅ *(meilleur modèle)*
- Naive Bayes
- Support Vector Machine (SVM)

> 🎯 **Précision atteinte :** 97% avec le modèle **Random Forest**

---

## 🧰 Fichiers Utiles

- `utils.py` : Fonctions pour transformer les labels à partir des noms de dossiers
- `plant_disease_classifier.ipynb` : Notebook principal pour l'entraînement et les tests
- `dataset/` : Dossier contenant les images classées en sous-dossiers `Healthy` et `Diseased`
- `features.hdf5` : Fichier de sauvegarde des caractéristiques extraites
- `test_notebook.ipynb` : Notebook dédié à l'évaluation des performances sur de nouvelles images

---

## 🚀 Pour commencer

1. Installe les dépendances :

```bash
pip install numpy opencv-python pillow scikit-learn mahotas h5py matplotlib
