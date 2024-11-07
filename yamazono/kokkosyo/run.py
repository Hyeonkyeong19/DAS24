import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


print('データの読み込みと前処理の実施…')
# データの読み込み
train_data = pd.read_csv('train.csv', low_memory=False)
test_data = pd.read_csv('test.csv', low_memory=False)

# 不要なカラムの削除（予測に役立たないと考えられるカラム）
columns_to_drop = [
    'building_id', 'building_name', 'building_name_ruby', 'homes_building_name', 'homes_building_name_ruby', 
    'full_address', 'unit_id', 'unit_name', 'name_ruby', 'snapshot_create_date', 'snapshot_modify_date', 
    'timelimit_date', 'post1', 'post2', 'addr1_1', 'addr1_2', 'addr2_name', 'addr3_name', 'addr4_name',
    'rosen_name1', 'eki_name1', 'bus_stop1', 'rosen_name2', 'eki_name2', 'bus_stop2', 'traffic_other',
    'traffic_car', 'school_ele_name', 'school_jun_name', 'est_other_name'
]
train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

# カテゴリカル変数のエンコーディング
categorical_columns = [
    'building_status', 'building_type', 'building_structure', 'building_land_chimoku', 'land_youto', 'land_toshi',
    'land_chisei', 'management_form', 'management_association_flg', 'bukken_type', 'flg_investment', 'genkyo_code',
    'usable_status', 'building_area_kind', 'land_area_kind', 'land_road_cond', 'land_seigen', 'land_setback_flg',
    'flg_open', 'flg_own', 'house_kanrinin', 'parking_kubun', 'parking_keiyaku'
]
le_dict = {}
for col in tqdm(categorical_columns, desc='カテゴリカル変数エンコーディング'): 
    if col in train_data.columns:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        le_dict[col] = le
        if col in test_data.columns:
            # Test data に未知のカテゴリが含まれている場合、未知のラベルを -1 として処理
            test_data[col] = test_data[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
            le.classes_ = np.append(le.classes_, '<unknown>')
            test_data[col] = le.transform(test_data[col].astype(str))

# 日付データの処理
date_columns = ['building_create_date', 'building_modify_date']
for col in date_columns:
    if col in train_data.columns:
        train_data[col] = pd.to_datetime(train_data[col], errors='coerce')
        train_data[col] = train_data[col].map(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    if col in test_data.columns:
        test_data[col] = pd.to_datetime(test_data[col], errors='coerce')
        test_data[col] = test_data[col].map(lambda x: x.timestamp() if pd.notnull(x) else np.nan)

# 学習用データの前処理
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# 数値変換可能な文字列を処理
for col in train_data.select_dtypes(include=['object']).columns:
    try:
        train_data[col] = pd.to_numeric(train_data[col].str.replace(',', '').str.replace(' ', ''), errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col].str.replace(',', '').str.replace(' ', ''), errors='coerce')
    except ValueError:
        pass

# 欠損値の再確認と処理
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# 目的変数（賃料）と説明変数の分割
y = train_data['money_room']
X = train_data.drop(['money_room'], axis=1)

# データを学習用と検証用に分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.reset_index(drop=True, inplace=True)
X_valid.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)

print('モデルの学習中…')
# 複数のモデルを使用したアンサンブル学習
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, random_state=42)
rf_model = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42)
ridge_model = Ridge(alpha=1.0)

# アンサンブルモデル（Voting Regressor）
ensemble_model = VotingRegressor(estimators=[('lgb', lgb_model)])

# モデルの学習
ensemble_model.fit(X_train, y_train)

# 検証用データで予測
y_pred = ensemble_model.predict(X_valid)

# モデルの評価（RMSEを使用）
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# 評価用データでの予測
test_data = test_data[X.columns]  # 学習時と同じ特徴量を選択
test_predictions = ensemble_model.predict(test_data)

# 予測結果の保存
submission = pd.read_csv('sample_submit.csv', header=None)
submission['money_room'] = test_predictions[:len(submission)]
submission = submission.iloc[:, [0, -1]]  # 最初の列と'money_room'の2列にする
submission.to_csv('submission.csv', index=False, header=False)
print('保存が完了しました。')