import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def plot_linear_reg(name, y_pred, X_test, y_test):
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.savefig(name)


def find_linearReg_optimal_test_size(X, y):
    scores = []
    for size in np.arange(0.05, 1, 0.05):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

        mdl = LinearRegression().fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        scores.append((size, r2_score(y_test, y_pred)))
    return scores[np.argmax([tp[1] for tp in scores])][0]


def check_outlier(x, scaled=True):
    if scaled:
        pass


def plot_scatter(x, y, name):
    plt.scatter(x, y, color='black', edgecolors='blue', alpha=0.25)

    plt.xticks(())
    plt.yticks(())

    plt.savefig(name)
