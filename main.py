import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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