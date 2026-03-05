# Compte Rendu - Régression Quadratique

## 1. Standardisation des Données
Avant de commencer, on a standardisé les données :
- Soustraire la moyenne
- Diviser par l'écart-type

**Pourquoi ?** Cela aide la descente de gradient à converger plus vite et rend les gradients stables.

## 2. Le Modèle Quadratique
Fonction de prédiction : $\hat{y} = ax^2 + bx + c$

C'est une parabole, plus flexible qu'une droite. Les poids à apprendre sont : $a$, $b$, $c$

## 3. Dérivation des Gradients

**Fonction de perte :**
$$L = \frac{1}{n} \sum_{i=1}^{n} (e_i)^2 \text{ où } e_i = \hat{y}_i - y_i$$

**Dérivées partielles :**
$$\frac{\partial L}{\partial a} = \frac{2}{n} \sum_{i=1}^{n} e_i x_i^2$$
$$\frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i x_i$$
$$\frac{\partial L}{\partial c} = \frac{2}{n} \sum_{i=1}^{n} e_i$$

Le facteur $\frac{1}{n}$ intervient parce qu'on moyenne sur toutes les données.

## 4. Backpropagation
Calcul des gradients et mise à jour des poids :
$$a \leftarrow a - \eta \frac{\partial L}{\partial a}$$
$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$
$$c \leftarrow c - \eta \frac{\partial L}{\partial c}$$

où $\eta$ est le learning rate.

**À quoi sert $\eta$ ?** Il contrôle la taille des pas. Petit $\eta$ = convergence lente, grand $\eta$ = risque de divergence.

## 5. Descente de Gradient
1. Initialiser $a$, $b$, $c$ aléatoirement
2. Pour chaque epoch :
   - Calculer les prédictions
   - Calculer les gradients
   - Mettre à jour les poids
   - Stocker la RMSE
3. Répéter jusqu'à convergence

**Symptômes d'un learning rate trop grand :**
- RMSE augmente
- Les poids explosent

**Symptôme d'un learning rate trop petit :**
- Convergence très lente

## 6. Résultats
Le code entraîne le modèle et affiche :
- Nuage de points + courbe quadratique prédite
- Évolution de la RMSE à chaque epoch
- RMSE final et paramètres du modèle

La RMSE devrait décroître globalement, mais peut avoir des petites variations.

