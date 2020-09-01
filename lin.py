import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from shared_utilities import find_linearReg_optimal_test_size
from shared_utilities import plot_linear_reg
from shared_utilities import check_outlier
from shared_utilities import plot_scatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.read_csv('Real estate.csv')

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    plot_scatter(X.iloc[:, 2], y, 'scatter.png')

    # Scale X
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    f = open("values.txt", "a")
    f.write(str(X))
    f.close()

    raise Exception()

    # PCA for feature reduction
    pca = PCA(n_components='mle', svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(X_pca.min(), X_pca.max())
    test_size = find_linearReg_optimal_test_size(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42)

    mdl = LinearRegression().fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    score = r2_score(y_test, y_pred)

    # plot_linear_reg('linear_reg.png', y_pred, X_test.iloc[:, 2], y_test)
