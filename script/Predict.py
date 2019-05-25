# Predict.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Value:

    """
         _split: データを訓練用と検証用、および評価用に分割し返す
           _fit: 学習させたモデルを返す
       _predict: 予測データを返す
       evaluate: RMSEで評価した結果を返す
    linear_plot: 学習させたモデルより近似直線をプロット
           main: 予測データを元に作成したDFを返す
    """

    def __init__(self, df, sweet=None, other=None, model=LinearRegression()):
        df = df.dropna(how='all', subset=[sweet, other])
        df_mod = df[df[other].isnull()==False]
        self._df = df
        self._df_mod = df_mod
        self._sweet = sweet
        self._other = other
        self._mod = model

    def _split(self, state='learning'):
        # データを訓練用、検証用、および評価用に分割する
        df_train = self._df_mod[(self._df_mod[self._sweet].isnull()==False)
                                &(self._df_mod[self._other].isnull()==False)]
        X, y = df_train.loc[:, df_train.columns!=self._sweet], df_train[self._sweet]
        if state=='learning':
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
            return X_train, y_train, X_valid, y_valid
        else:
            df_test = self._df_mod[(self._df_mod[self._sweet].isnull()==True)
                                   &(self._df_mod[self._other].isnull()==False)]
            X_test = df_test.loc[:, df_test.columns!=self._sweet]
            return X, y, X_test

    def _fit(self, X, y):
        # モデルを学習させる
        return self._mod.fit(X, y)

    def _predict(self, mod, X):
        # モデルより予測データを求める
        return mod.predict(X)

    def evaluate(self):
        # モデルをRMSEで評価する
        X_train, y_train, X_valid, y_valid = self._split(state='learning')
        mod = self._fit(X_train, y_train)
        y_pred = self._predict(mod, X_valid)
        return np.sqrt(mean_squared_error(y_valid, y_pred))

    def main(self, col):
        # 予測データを求め、DFを作成する
        X_train, y_train, X_test = self._split(state='predicting')
        mod = self._fit(X_train, y_train)
        y_test = self._predict(mod, X_test)
        df_pred = pd.DataFrame(index=self._df.index)
        df_pred[col] = self._df[self._sweet]
        df_pred[col].loc[df_pred[col].isnull()] = y_test
        return df_pred


class Variance:

    """
       _conversion_rate: 第２主成分の固有ベクトルから求めた変換率を返す
    _secondary_variance: 第２主成分方向の分散を返す
                   main: 予測データの誤差を返す
    """

    def __init__(self, df, sweet=None, other=None, dsweet=None, n_components=2):
        df_tar = df.dropna(subset=[sweet, other[0]])
        X = df_tar[sweet].values
        for col in other:
            X = np.vstack([X, df_tar[col]])
        self._df = df
        self._df_tar = df_tar
        self._dsweet = dsweet
        self._X = X.T
        self._pca = PCA(n_components=n_components)

    def _conversion_rate(self):
        # 第２主成分方向の固有ベクトルから変換率を求める
        eigen_size = np.sqrt(np.sum(self._pca.components_[1, :]**2))
        return np.abs(self._pca.components_[1, 0])/eigen_size
        
    def _secondary_variance(self):
        # 第１主成分方向をx軸に指定し、第２主成分方向の分散を求める
        Xd = self._pca.transform(self._X)
        return Xd[:, 1].std()

    def main(self, col):
        # ２次元データを学習し、予測データの誤差を求める
        self._pca.fit(self._X)
        rate = self._conversion_rate()
        sigma_2 = self._secondary_variance()
        df_pred = pd.DataFrame(index=self._df.index)
        df_pred[col] = self._df[self._dsweet]
        df_pred[col].loc[df_pred[col].isnull()] = sigma_2*rate
        return df_pred

