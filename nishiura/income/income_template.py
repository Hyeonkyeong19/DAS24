import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import warnings
import pdb
warnings.filterwarnings('ignore')

path = "/Users/remi/dataAnalysis/dataAnalysis2022/preprocessing/income/dataset/"

df = pd.read_csv(path + 'train.tsv', delimiter='\t')
df_test = pd.read_csv(path + 'test.tsv', delimiter='\t')

pdb.set_trace()

## -------------------------------------------------
## データの前処理


## -------------------------------------------------

pdb.set_trace()

## ベースラインモデルの構築
y = df.loc[:, "Y"].values
X = df.drop("Y", axis=1).iloc[:,1:].values

X_test = df_test.iloc[:, 1:].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

rfc = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=100, n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)
print('Random Forest')
print('Trian Score: {}'.format(round(rfc.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(rfc.score(X_valid, y_valid), 3)))

## モデルのアンサンブリング
rfc_pred = rfc.predict_proba(X_test)
pred = rfc_pred.argmax(axis=1)

'''
## 提出
submission = pd.read_csv(path + 'sample_submit.csv', header=None)
submission[1] = pred
submission.to_csv(path + 'submission.csv', index=False, header=None)
'''