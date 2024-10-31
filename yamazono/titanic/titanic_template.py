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
pd.set_option('display.max_columns', 150)

path = "/home/yamazono/DAS24/yamazono/titanic/dataset/"

df = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')


## -------------------------------------------------
## データの前処理
# 必要なカテゴリ変数をエンコード
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # 性別のエンコード
df['Embarked'].fillna('S', inplace=True)  # 欠損値を補完
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # 出発港のエンコード

# 不要な列を削除
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 同様にテストデータも処理
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})
df_test['Embarked'].fillna('S', inplace=True)
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 欠損値を補完（例として年齢を中央値で補完）
df['Age'].fillna(df['Age'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

print(df.columns)

# 特徴量とラベルを再定義
X = df.drop('Perished', axis=1).values  # 'Survived' ではなく 'Perished' を使用
y = df['Perished'].values
X_test = df_test.values

## -------------------------------------------------

## ベースラインモデルの構築
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

X_test = df_test.iloc[:, 1:].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

rfc = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=100, n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)

print('Random Forest')
print('Trian Score: {}'.format(round(rfc.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(rfc.score(X_valid, y_valid), 3)))

## 様々なモデルの構築・調整
### ロジスティック回帰モデル
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

print('Logistic Regression')
print('Train Score: {}'.format(round(lr.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(lr.score(X_valid, y_valid), 3)))

### 多層パーセプトロンモデル
mlpc = MLPClassifier(hidden_layer_sizes=(100, 100, 10), random_state=0)
mlpc.fit(X_train, y_train)

print('Multilayer Perceptron')
print('Train Score: {}'.format(round(mlpc.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(mlpc.score(X_valid, y_valid), 3)))

## モデルのアンサンブリング
rfc_pred = rfc.predict_proba(X_test)
lr_pred = lr.predict_proba(X_test)
mlpc_pred = mlpc.predict_proba(X_test)
pred_proba = (rfc_pred + lr_pred + mlpc_pred) / 3
pred = pred_proba.argmax(axis=1)

'''
## 提出
submission = pd.read_csv(path + 'gender_submission.csv')
submission['Perished'] = pred
submission.to_csv(path + 'submission.csv', index=False)
'''
