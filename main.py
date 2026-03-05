import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def quadratic_regression(a, b, c, x):
    return a * x**2 + b * x + c

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))

def grad_a(y_pred, y_true, x):
    error = y_pred - y_true
    return 2 * np.mean(error * x**2)

def grad_b(y_pred, y_true, x):
    error = y_pred - y_true
    return 2 * np.mean(error * x)

def grad_c(y_pred, y_true):
    error = y_pred - y_true
    return 2 * np.mean(error)
def backpropagation_quadratic(x, y, a, b, c, learning_rate):
    y_pred = quadratic_regression(a, b, c, x)
    
    dL_da = grad_a(y_pred, y, x)
    dL_db = grad_b(y_pred, y, x)
    dL_dc = grad_c(y_pred, y)
    
    a = a - learning_rate * dL_da
    b = b - learning_rate * dL_db
    c = c - learning_rate * dL_dc
    
    error_rmse = rmse(y_pred, y)
    
    return a, b, c, error_rmse

def gradient_descent_quadratic(x, y, epochs, learning_rate):
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    
    rmse_history = []
    
    for i in range(epochs):
        a, b, c, error = backpropagation_quadratic(x, y, a, b, c, learning_rate)
        rmse_history.append(error)
    
    return a, b, c, rmse_history
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
    
    x = house_prices_df["surface"].values
    y = house_prices_df["prix"].values
    
    print(house_prices_df.head(5))
    print(house_prices_df.dtypes)
    print(house_prices_df.shape)
    
    a, b, c, rmse_history = gradient_descent_quadratic(x, y, epochs=1000, learning_rate=0.01)
    
    y_pred = quadratic_regression(a, b, c, x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.5)
    x_sorted = np.sort(x)
    y_sorted = quadratic_regression(a, b, c, x_sorted)
    plt.plot(x_sorted, y_sorted, color='red', label='prediction')
    plt.xlabel("surface")
    plt.ylabel("prix")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(rmse_history)
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    
    plt.tight_layout()
    plt.show()
    
    print(f"RMSE final: {rmse_history[-1]}")
    print(f"Parametres: a={a}, b={b}, c={c}")
    plt.show()

    