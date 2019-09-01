# Simulation_v19.py

import numpy as np
from numpy import random as rd
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt



class Simulation:

    """
    __init__        : パラメータの初期化
    value_set       : 正規分布を仮定した値を返す
    limit_value_set : 検出限界の起動長半径と惑星質量を返す
    judgement       : ランダムに選択した惑星が検出できるかを判定し、検出できたものを返す
    ks_test         : 2標本の分布をKS検定で評価し、その評価値を返す
    ad_test         : 2標本の分布をAD検定で評価し、その評価値を返す
    GMM             : Gaussian Mixture Modelで学習したモデルを返す
    """

    def __init__(self, df, metal='rich', rich_border=0.0, poor_border=-0.1, random_state=None):
        
        # ランダムの状態を設定
        self._random_state = random_state
        
        # 金属量の多い領域と低い領域にDFを分割
        _df_rich = df[df['Fe/H']>=rich_border]
        _df_poor = df[df['Fe/H']<poor_border]
        
        # 予測DFと検出限界DFを設定
        if metal=='rich':
            df_obs = self.value_set(_df_rich)
            df_lim = self.limit_value_set(_df_poor)
        elif metal=='poor':
            df_obs = self.value_set(_df_poor)
            df_lim = self.limit_value_set(_df_rich)
        self.df_obs = df_obs
        self._df_lim = df_lim
        
        
    def value_set(self, df):
        
        # 観測結果より予測データを作成
        if not self._random_state==0:
            a = rd.normal(df['a'], df['da'])
            Mp = rd.normal(df['Mp'], df['dMp'])
            e = rd.normal(df['e'], df['de'])
            Fe_H = rd.normal(df['Fe/H'], df['dFe/H'])
            e = np.where(e<0, 0, e)
            e = np.where(e>1, 1, e)
        else:
            a = df['a']
            Mp = df['Mp']
            e = df['e']
            Fe_H = df['Fe/H']
        _df = pd.DataFrame({'a': a, 'da': df['da'],
                            'Mp': Mp, 'dMp': df['dMp'],
                            'e': e, 'de': df['de'],
                            'Fe/H': Fe_H, 'dFe/H': df['dFe/H'],
                            'Ms': df['Ms'], 'dMs': df['dMs']
                           })
        
        return _df

    
    def limit_value_set(self, df):
        
        # 各パラメータを配列として抽出
        if not self._random_state==0:
            Ms = rd.normal(df['Ms'].values, df['dMs'].values)
            e = rd.normal(df['e'].values, df['de'].values)
            e = np.where(e<0, 0, e)
            e = np.where(e>1, 1, e)
        else:
            Ms = df['Ms']
            e = df['e']
        RMS = df['RMS'].values
        Term = df['Term'].values
        
        # 検出精度、観測期間より検出限界を計算
        a_lim = Term**(2/3)*Ms**(1/3)
        Mp_lim = 0.004919*RMS*(1-e**2)**(1/2)*Ms**(2/3)*(Term*365)**(1/3)
        
        # 検出限界のDFを作成
        _df = pd.DataFrame({'a_lim': a_lim, 'Mp_lim': Mp_lim, 'RMS': RMS, 'Term': Term})
        
        # データが欠損している行を削除
        _df = _df.dropna(subset=['a_lim', 'Mp_lim'], how='any')
        
        return _df

    
    def judgement(self):
        
        # 予測DFと検出限界DFからランダムにサンプルを取得
        n = len(self.df_obs)
        df_tar = self.df_obs.sample(n=n, random_state=self._random_state).reset_index(drop=False)
        df_lim = self._df_lim.sample(n=n, replace=True, random_state=self._random_state).reset_index(drop=True)
        df_merge = df_tar.join(df_lim).set_index(['star', 'planet'])
        
        # 検出限界を満たす予測データを抽出
        a_tar = df_merge['a']
        Mp_tar = df_merge['Mp']
        a_lim = df_merge['a_lim']
        Mp_lim = df_merge['Mp_lim']
        df_judge = df_merge[(a_tar/a_lim<=(Mp_tar/Mp_lim)**2)&(a_tar/a_lim<=1)].drop(columns=['a_lim', 'Mp_lim'])
                
        return df_judge
    
    
    def ks_test(self, data1, data2):
        
        # KS検定で最大離隔率とp値を取得
        ks = ks_2samp(data1, data2)
        statistic = ks.statistic
        pvalue = ks.pvalue
        
        #return statistic, pvalue
        return pvalue
    
    
    def ad_test(self, data1, data2):
        
        # AD検定で最大離隔率とCV、およびp値を取得
        ad = anderson_ksamp([data1, data2])
        statistic = ad.statistic
        cv = ad.critical_values
        pvalue = ad.significance_level
        
        #return statistic, cv, pvalue
        return pvalue
    
    
    def GMM(self, X, n_components=2, covariance_type='full'):
        
        # Gussian Mixture Modelを設定
        gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=self._random_state)
        
        # 訓練データを入力し、学習モデルを作成
        gm.fit(X)
        
        return gm

    
    
class Extraction:
    
    """
    __init__        : パラメータの初期化
    data_extraction : 金属量の多い領域と少ない領域、および全域の予測データを配列データとして返す
    _extraction     : Simulationのjudgementで得た予測データを配列データとして返す
    AIC_extraction  : 予測データをクラスター数を変えてGMMで学習し、AICで評価した値を返す
    BIC_extraction  : 予測データをクラスター数を変えてGMMで学習し、BICで評価した値を返す
    data_division   : 惑星質量境界もしくはラベルでデータを分割し、リストとして返す
    """

    def __init__(self, df, random_state=None):
        
        # パラメータをランダムに振るかを指定
        self._random_state = random_state
        
        # 観測DFを設定
        self._df = df
        

    def data_extraction(self, rich_border=0.0, poor_border=-0.1):
        
        # 金属量の境界と抽出サンプル数を設定
        self._rich = rich_border
        self._poor = poor_border

        # 金属量の領域に応じて予測DFを作成
        df_rich = self._extraction('rich')
        df_rich['metal'] = 'rich'
        df_poor = self._extraction('poor')
        df_poor['metal'] = 'poor'
        df_all = df_rich.append(df_poor).sort_index()
        
        return df_all
        
        
    def _extraction(self, metal):
        
        # Simulation classを設定
        sim = Simulation(self._df, metal=metal, rich_border=self._rich, poor_border=self._poor, random_state=self._random_state)
        
        # 検出判定したパラメータを抽出
        df_judge = sim.judgement()
        
        return df_judge
    
    
    def AIC_extraction(self, X, n_components=10, covariance_type='full'):
        
        # GMMで学習したモデルをAkaike's Information Criterionで評価
        AIC = []
        for n in range(n_components):
            gm = GaussianMixture(n_components=n+1, covariance_type=covariance_type, random_state=self._random_state)
            gm.fit(X)
            AIC.append(gm.aic(X))
            
        return AIC
    
    
    def BIC_extraction(self, X, n_components=10, covariance_type='full'):
        
        # GMMで学習したモデルをBayesian Information Criterionで評価
        BIC = []
        for n in range(n_components):
            gm = GaussianMixture(n_components=n+1, covariance_type=covariance_type, random_state=self._random_state)
            gm.fit(X)
            BIC.append(gm.bic(X))
            
        return BIC

    
    def data_division(self, df, param, limit_a=None, borders=None, labels=None):
        
        # 軌道長半径が0.1以上の惑星だけを抽出
        if limit_a==True: _df = df[df['a']>=0.1]
        else: _df = df
        
        # 惑星の境界質量を設定し分割
        if borders:
            cluster1 = _df[_df['Mp']<borders[0]][param]
            cluster2 = _df[(_df['Mp']>=borders[0])&(_df['Mp']<borders[1])][param]
            cluster3 = _df[_df['Mp']>=borders[1]][param]
            
        # GMMで得たラベルを元に分割
        elif labels:
            cluster1 = _df[_df['label_gm']==labels[0]][param]
            cluster2 = _df[_df['label_gm']==labels[1]][param]
            cluster3 = _df[_df['label_gm']==labels[2]][param]
        
        return [cluster1, cluster2, cluster3]

    

class Plot:
    
    """
    __init__          : パラメータの初期化
    pvalues_plot      : 各金属量の境界値におけるp値を図示する
    histogram         : ヒストグラムおよび累積分布を図示する
    mass_eccentricity : 惑星質量と軌道離心率の散布図を図示する
    gmm_plot          : GMMで学習した場合のクラスター数ごとの評価を図示する
    labels_plot       : 最適なクラスター数を用いてGMMで分類し、軌道長半径と惑星質量の散布図を図示する
    
    """
    
    def __init__(self, fig):
        
        # 図のベースを設定
        self.fig = fig
    
    
    def pvalues_plot(self, ax, borders=None, pvalues=None, pvalues_err=None, title=None):
        
        # 各金属量の境界値におけるp値を誤差棒付きの散布図で図示
        ax.plot(borders, pvalues, 'ro', color='r')
        ax.errorbar(borders, pvalues, yerr=pvalues_err, fmt='ro', ecolor='k', ms=0)
        ax.grid(which='major', linestyle='dashed')
        
        # 図の詳細を設定
        if title: ax.set_title(title)
        ax.set_yscale('log')
        ax.set_xlabel('Boundary of Metallicity [Fe/H]')
        ax.set_ylabel('p-value')
        
    
    def histogram(self, ax, data, bins=20, logx=False, cumulative=False, normed=False, alpha=0.5, label=None, loc='best', title=None, xlabel=None, ylabel=True, xlim=None):
        
        # データの色を設定
        colors = ['r', 'g', 'b', 'm', 'c']
        
        # x軸が対数表示の場合の設定
        if logx:
            bins = [10**np.linspace(np.floor(np.log10(np.min(x))), np.ceil(np.log10(np.max(x))), bins) for x in data]
            ax.set_xscale('log')
        else: bins = [np.linspace(0, 1, bins)]*len(data)
        
        # ラベルがなければNoneのリストを作成
        if not label:
            label = [None for _ in range(len(data))]
        
        # 入力データのヒストグラムor累積分布を図示
        for n, (_data, _bins, _label) in enumerate(zip(data, bins, label)):
            ax.hist(_data, bins=_bins, histtype='barstacked', cumulative=cumulative, normed=normed, alpha=alpha, color=colors[n], label=_label)
        
        # 図の詳細を設定
        if title: ax.set_title(title)
        if label[0]: ax.legend(loc=loc)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel:
            if cumulative: ax.set_ylabel('Cumulative Fraction')
            else: ax.set_ylabel('Number of Planets')
        if xlim: ax.set_xlim(right=xlim)
            
    
    def mass_semi(self, ax, a, Mp, borders=None, title=None, ylabel=True):
        
        # データの色を設定
        colors = ['r', 'g', 'b']
        
        # 領域ごとのデータを散布図で図示
        [ax.scatter(x, y, s=20, c=c) for x, y, c in zip(a, Mp, colors)]
        
        if borders: [ax.hlines([bord], -0.1, 6, 'black', linestyles='dashed') for bord in borders]
        
        # 図の詳細を設定
        ax.loglog()
        if title: ax.set_title(title)
        ax.set_xlabel('Semi-Major Axis (au)')
        if ylabel: ax.set_ylabel('Lower Limit of Companion Mass [$\it{M_J}$]')
        ax.set_xlim(-0.05, 5.9)
        ax.set_ylim(0.07, 200)
        ax.tick_params(labelleft=ylabel)
        
            
    def mass_eccentricity(self, ax, e, Mp, borders=None, title=None, ylabel=True):
        
        # データの色を設定
        colors = ['r', 'g', 'b']
        
        # 領域ごとのデータを散布図で図示
        [ax.scatter(x, y, s=20, c=c) for x, y, c in zip(e, Mp, colors)]
        
        if borders: [ax.hlines([bord], -0.1, 1.1, 'black', linestyles='dashed') for bord in borders]
        
        # 図の詳細を設定
        ax.set_yscale('log')
        if title: ax.set_title(title)
        ax.set_xlabel('Eccentricity')
        if ylabel: ax.set_ylabel('Lower Limit of Companion Mass [$\it{M_J}$]')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.07, 200)
        ax.tick_params(labelleft=ylabel)
    
    
    def gmm_plot(self, i=111, n_components=None, AIC=None, BIC=None, title=None, save_name=None):
        
        # i番目の位置にサブプロットするよう設定
        ax = self.fig.add_subplot(i)
        
        # GMMで学習したモデルをAICで評価した結果を点付き折れ線グラフで表示
        try: ax.plot(n_components, AIC, '--bo', c='r', label='AIC')
        except: pass
            
        # GMMで学習したモデルをBICで評価した結果を点付き折れ線グラフで表示
        try: ax.plot(n_components, BIC, '--bo', c='b', label='BIC')
        except: pass
            
        # 図の詳細を設定
        ax.legend(loc='upper center', ncol=2)
        if title: ax.set_title(title)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Error Score')
        ax.set_xticks(n_components)
        
     
    def labels_plot(self, ax, df, xdata=None, borders=None, title=None, labely=False):
        
        # プロットのデータを取得
        label = df['label'].unique()
        if xdata=='metal': x = [df[df['label']==l]['Fe/H'].values for l in label]
        elif xdata=='a': x = [df[df['label']==l]['a'].values for l in label]
        y = [df[df['label']==l]['Mp'].values for l in label]
        color = [df[df['label']==l]['color'].values for l in label]
        if labely: marker = ['o', '^', ',']
        else: marker = ['^', 'o', ',']
        
        # 領域で分けた状態を色に、ラベル分けした状態を記号に反映させて散布図を図示
        for _x, _y, _c , _m in zip(x, y, color, marker):
            ax.scatter(_x, _y, c=_c, marker=_m, s=30)
        
        # 図の詳細を設定
        ax.set_yscale('log')
        if borders: [ax.hlines([bord], -1.0, 10, 'black', linestyles='dashed') for bord in borders]
        if title: ax.set_title(title)
        if labely: ax.set_ylabel('Lower Limit of Companion Mass ($M_J$)')
        if xdata=='metal':
            ax.set_xlabel('Host-star Metalicity [Fe/H] (dex)')
            ax.set_xlim(-1.0, 0.5)
        elif xdata=='a':
            ax.set_xlabel('Semi-Major Axis (au)')
            #ax.set_xlim(0, 5.9)
        ax.set_ylim(0.05, 200)
        
