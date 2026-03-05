import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quadratic_regression(a, b, c, x):
    return a * x**2 + b * x + c
# c'est plus flexible qu'un modèle affine car il peut s'adapter  a plusieurs types de données et en plus c'est une parabole 
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))
#Quelle différence d’interprétation entre MSE et RMSE ?
# les unité de la mse sont au carré et pour la rmse c'est la même unité que les données
#Pourquoi la RMSE est parfois plus lisible ?
#Parce que RMSE est dans les mêmes unités que les données.


if __name__ == "__main__":
    house_prices_df = pd.read_csv("prix_maisons.csv")
    x_mean, x_std= house_prices_df["surface"].mean(), house_prices_df["surface"].std()
    y_mean, y_std = house_prices_df["prix"].mean(), house_prices_df["prix"].std()

    house_prices_df["surface"] = (house_prices_df["surface"] - x_mean) / x_std
    house_prices_df["prix"] = (house_prices_df["prix"] - y_mean) / y_std
    
    print(house_prices_df.head(5))
    print(house_prices_df.dtypes)
    print(house_prices_df.shape)
    #Questions
    # Pourquoi la standardisation aide la descente de gradient ?
    #La standardisation aide la descente de gradient parce qu'elle rend toutes les variables à la même échelle du coup sa converge plus vite 

    #     #Que se passe-t-il si on ne normalise pas et que les surfaces sont en dizaines alors que les prix
    # sont en centaines de milliers ?
    #sa vas converger de manière très lente le gradient pour le prix sera énorme 

    plt.scatter(house_prices_df["surface"], house_prices_df["prix"])
    plt.xlabel("surface")
    plt.ylabel("prix")
    plt.show()

    