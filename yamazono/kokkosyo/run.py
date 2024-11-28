import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import BallTree
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import lightgbm as lgb
import os
import pdb

print('データの読み込みと前処理の実施…')
# データの読み込み
train_data = pd.read_csv('train.csv', low_memory=False)
test_data = pd.read_csv('test.csv', low_memory=False)

train_data = train_data[0:100]
test_data = test_data[0:20]

# 逆距離加重法 (IDW) による空間補間を行う関数
def idw_interpolation(gdf_target, gdf_source, value_column, k=10, power=2):
    """
    逆距離加重法 (IDW) による空間補間を行う関数
    gdf_target: 補間対象の GeoDataFrame（物件データ）
    gdf_source: 補間元の GeoDataFrame（地価データなど）
    value_column: 補間に使用する値のカラム名
    k: 使用する最も近いポイントの数
    power: 距離の重みづけの際の指数
    """
    # ポイント以外のジオメトリが含まれていないかチェック
    gdf_source = gdf_source[gdf_source.geometry.type == 'Point']
    if gdf_source.empty:
        raise ValueError("IDW interpolation requires at least one Point geometry in the source GeoDataFrame.")
    if not all(gdf_target.geometry.type == 'Point'):
        raise ValueError("IDW interpolation requires all target geometries to be Points.")
    
    target_coords = np.array([gdf_target.geometry.x, gdf_target.geometry.y]).T
    source_coords = np.array([gdf_source.geometry.x, gdf_source.geometry.y]).T
    source_values = gdf_source[value_column].values

    tree = BallTree(source_coords, leaf_size=15)
    distances, indices = tree.query(target_coords, k=k)

    weights = 1 / (distances ** power)
    weights[np.isinf(weights)] = 0

    interpolated_values = np.sum(weights * source_values[indices], axis=1) / np.sum(weights, axis=1)
    return interpolated_values

# 国土数値情報の読み込みとマージ
print('国土数値情報の読み込みとマージ…')
land_value_data_path = os.path.join('data', 'L02-24_GML', 'L02-24.shp')
land_value_data = gpd.read_file(land_value_data_path)
land_data_path = os.path.join('data', '公示地価_神奈川', 'L01-24_14.shp')
land_data = gpd.read_file(land_data_path)  # 公示地価データの読み込み
land_data = land_data[['geometry', 'L01_006']]  # 必要なカラムを指定（L01_006は公示地価）

# 緯度・経度をキーにして、追加データを物件データにマージ
train_geo = gpd.GeoDataFrame(train_data, geometry=gpd.points_from_xy(train_data['lon'], train_data['lat']), crs='EPSG:4326')
test_geo = gpd.GeoDataFrame(test_data, geometry=gpd.points_from_xy(test_data['lon'], test_data['lat']), crs='EPSG:4326')

# CRSの変換
land_data = land_data.to_crs(train_geo.crs)

# 地理空間情報とマージ（空間結合）
train_data = gpd.sjoin(train_geo, land_data, how='left', predicate='intersects')
test_data = gpd.sjoin(test_geo, land_data, how='left', predicate='intersects')

# 必要な情報のみ保持
train_data = pd.DataFrame(train_data.drop(columns='geometry'))
test_data = pd.DataFrame(test_data.drop(columns='geometry'))

# 地価情報を物件データに追加
train_geo['land_value_idw'] = idw_interpolation(train_geo, land_value_data, 'L02_006')
test_geo['land_value_idw'] = idw_interpolation(test_geo, land_value_data, 'L02_006')

# 駅別乗降客数データの読み込みと逆距離加重法の適用
print('駅別乗降客数データの読み込みと逆距離加重法による補間…')
station_data_path = os.path.join('data', 'S12-23_GML', 'S12-23_NumberOfPassengers.shp')
station_data = gpd.read_file(station_data_path)

# LINESTRINGから代表ポイント（開始点）を抽出
if not all(station_data.geometry.type == 'Point'):
    station_data['geometry'] = station_data.geometry.apply(lambda geom: geom.representative_point() if geom.type == 'LineString' else geom)

# 数値データ列を選択（仮にS12_053を使用する）
station_data = station_data[['geometry', 'S12_053']]  # 必要なカラムを指定（S12_053は乗降客数）

# CRSの変換
station_data = station_data.to_crs(train_geo.crs)

# 乗降客数情報を物件データに追加
train_geo['station_passengers_idw'] = idw_interpolation(train_geo, station_data, 'S12_053')
test_geo['station_passengers_idw'] = idw_interpolation(test_geo, station_data, 'S12_053')

# geometry カラムを削除し、データフレームに戻す
train_data = pd.DataFrame(train_geo.drop(columns='geometry'))
test_data = pd.DataFrame(test_geo.drop(columns='geometry'))

# 駅別乗降客数、人口推移、公示地価、洪水浸水想定区域のデータもマージ
population_data_path = os.path.join('data', '人口推移', '500m_mesh_2018_14.shp')
flood_data_path = os.path.join('data', '洪水浸水想定区域_計画規模', 'A31b-10-23_10_5339.shp')

population_data = gpd.read_file(population_data_path)[['geometry']]  # 人口データの読み込み
flood_data = gpd.read_file(flood_data_path)[['geometry']]  # 洪水浸水想定区域のデータの読み込み

# CRSの変換
population_data = population_data.to_crs(train_geo.crs)
flood_data = flood_data.to_crs(train_geo.crs)

# 地理空間情報とマージ（空間結合）
train_data = gpd.sjoin(train_geo, population_data, how='left', predicate='intersects')
train_data = gpd.sjoin(train_geo, flood_data, how='left', predicate='intersects')

test_data = gpd.sjoin(test_geo, population_data, how='left', predicate='intersects')
test_data = gpd.sjoin(test_geo, flood_data, how='left', predicate='intersects')

# 必要な情報のみ保持
train_data = pd.DataFrame(train_data.drop(columns='geometry'))
test_data = pd.DataFrame(test_data.drop(columns='geometry'))

# 不要なカラムの削除（予測に役立たないと考えられるカラム）
columns_to_drop = [
    'building_id', 'building_name_ruby', 'homes_building_name', 'homes_building_name_ruby', 
    'full_address', 'unit_id', 'unit_name', 'name_ruby', 'snapshot_create_date', 'snapshot_modify_date', 
    'timelimit_date', 'post1', 'post2', 'addr1_1', 'addr1_2', 'addr2_name', 'addr3_name', 'addr4_name',
    'rosen_name1', 'eki_name1', 'bus_stop1', 'rosen_name2', 'eki_name2', 'bus_stop2', 'traffic_other',
    'traffic_car', 'school_ele_name', 'school_jun_name', 'est_other_name'
]
train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

# 年と月の分割と三角関数で特徴量を作成
def process_year_month(df, column):
    df['year'] = df[column] // 100
    df['month'] = df[column] % 100
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df.drop(columns=[column], inplace=True)

process_year_month(train_data, 'target_ym')
process_year_month(test_data, 'target_ym')

# 真値 (money_room) の分布確認（外れ値を無視して描画）
y = train_data['money_room']
y_filtered = y[(y > 0) & (y < y.quantile(0.99))]  # 外れ値を除去（0より大きく、上位1%を除外）
plt.hist(y_filtered, bins=50, edgecolor='k')
plt.xlabel('Rent Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Rent Amount (Filtered)')
plt.savefig('analysis/money_room_filtered.png')
plt.close()

# 賃料の対数変換
y = np.log1p(train_data['money_room'])  # 対数変換によりスケールを調整
train_data.drop(['money_room'], axis=1, inplace=True)

# unit_count の NaN 処理（中央値で補完）
train_data['unit_count'].fillna(train_data['unit_count'].median(), inplace=True)
test_data['unit_count'].fillna(train_data['unit_count'].median(), inplace=True)

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

# year_built の正規化
year_min = train_data['year_built'].min()
train_data['year_built_normalized'] = train_data['year_built'] - year_min
test_data['year_built_normalized'] = test_data['year_built'] - year_min

# 目的変数と説明変数の分割
X = train_data

# データを学習用と検証用に分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.reset_index(drop=True, inplace=True)
X_valid.reset_index(drop=True, inplace=True)

print('モデルの学習中…')
# LightGBM、GradientBoosting、RandomForest を使用したアンサンブル学習
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, num_leaves=64, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, random_state=42)
rf_model = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42)

# モデルの学習
for model, model_name in zip([lgb_model, gb_model, rf_model], ['LightGBM', 'GradientBoosting', 'RandomForest']):
    for i in tqdm(range(1), desc=f'モデルの学習中 ({model_name})'):
        model.fit(X_train, y_train)

# 検証用データで予測
y_pred_lgb = lgb_model.predict(X_valid)
y_pred_gb = gb_model.predict(X_valid)
y_pred_rf = rf_model.predict(X_valid)

# アンサンブルによる予測
y_pred_ensemble = (y_pred_lgb + y_pred_gb + y_pred_rf) / 3

# モデルの評価（RMSEを使用）
mse = mean_squared_error(y_valid, y_pred_ensemble)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# 対数変換を元に戻してRMSEを計算する場合（オプション）
y_valid_exp = np.expm1(y_valid)  # 対数変換を元に戻す
y_pred_ensemble_exp = np.expm1(y_pred_ensemble)  # 予測値も元に戻す
mse_original = mean_squared_error(y_valid_exp, y_pred_ensemble_exp)
rmse_original = np.sqrt(mse_original)
print(f'Root Mean Squared Error (original scale): {rmse_original}')

# 評価用データでの予測
test_data = test_data[X.columns]  # 学習時と同じ特徴量を選択
test_predictions_lgb = lgb_model.predict(test_data)
test_predictions_gb = gb_model.predict(test_data)
test_predictions_rf = rf_model.predict(test_data)

# アンサンブルによる評価用データの予測
test_predictions_ensemble = (test_predictions_lgb + test_predictions_gb + test_predictions_rf) / 3

# 予測結果の保存
submission = pd.read_csv('sample_submit.csv', header=None)
submission['money_room'] = np.expm1(test_predictions_ensemble[:len(submission)])  # 対数変換を戻す
submission = submission.iloc[:, [0, -1]]  # 最初の列と'money_room'の2列にする
submission.to_csv('submission.csv', index=False, header=False)
print('保存が完了しました。')
