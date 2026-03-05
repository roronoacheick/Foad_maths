# Compte Rendu - Régression Quadratique

## Objectif
Prédire les prix des maisons en fonction de leur surface en utilisant un modèle de régression quadratique.

## 1. Standardisation des Données
Avant de commencer, on a standardisé (normalisé) les données. Cela signifie :
- Soustraire la moyenne de chaque valeur
- Diviser par l'écart-type

**Pourquoi ?** Cela aide la descente de gradient à converger plus vite. Sans standardisation, les gradients seraient énormes et instables.

## 2. Le Modèle
On utilise une fonction quadratique pour faire les prédictions :

$$\hat{y} = ax^2 + bx + c$$

C'est une parabole, plus flexible qu'une droite. Elle peut s'adapter à plus de types de données.

**Les poids du modèle :** $a$, $b$, $c$
Ce sont les paramètres que le modèle doit apprendre.

## 3. Les Métriques d'Erreur
On mesure la qualité du modèle avec deux métriques :

**MSE (Mean Squared Error) :**
$$L = \frac{1}{n} \sum_{i=1}^{n} (e_i)^2$$
- Calcule la moyenne des erreurs au carré
- Unités au carré (difficile à interpréter)

**RMSE (Root Mean Squared Error) :**
$$RMSE = \sqrt{MSE}$$
- Racine carrée du MSE
- Unités identiques aux données (facile à interpréter)

## 4. Les Gradients
Les gradients mesurent comment l'erreur change quand on modifie les poids. Ils indiquent la direction pour améliorer le modèle.

$$\frac{\partial L}{\partial a} = \frac{2}{n} \sum_{i=1}^{n} e_i x_i^2$$

$$\frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i x_i$$

$$\frac{\partial L}{\partial c} = \frac{2}{n} \sum_{i=1}^{n} e_i$$

où $e_i = \hat{y}_i - y_i$ est l'erreur.

## 5. Comment ça Marche ?
1. On initialise les poids aléatoirement
2. On calcule les prédictions avec la formule quadratique
3. On mesure l'erreur avec MSE
4. On calcule les gradients
5. On met à jour les poids dans la direction opposée au gradient
6. On répète jusqu'à convergence

Cela s'appelle la **descente de gradient**.

## Conclusion
Ce modèle permet de prédire les prix des maisons de manière plus flexible qu'une simple droite. La standardisation et les gradients sont essentiels pour que le modèle apprenne correctement.
