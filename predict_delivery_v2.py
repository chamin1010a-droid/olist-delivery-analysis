"""
배송 예측 개선: 경로별 맞춤 버퍼 + 분위 회귀

지연율을 Olist 수준(5%)으로 낮추되, 예측 정확도는 유지하는 것이 목표.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / "ecommerce.db"
conn = sqlite3.connect(DB_PATH)

# ============================================================
# 1. 데이터 준비 (동일)
# ============================================================
df = pd.read_sql("""
    SELECT 
        o.order_id,
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date,
        o.order_delivered_customer_date,
        c.customer_state,
        s.seller_state,
        COALESCE(ct.product_category_name_english, p.product_category_name) as category,
        p.product_weight_g,
        p.product_length_cm,
        p.product_height_cm,
        p.product_width_cm,
        p.product_photos_qty,
        oi.price,
        oi.freight_value
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      AND o.order_status = 'delivered'
""", conn)
conn.close()

df['purchase_date'] = pd.to_datetime(df['order_purchase_timestamp'])
df['estimated_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['delivered_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['actual_days'] = (df['delivered_date'] - df['purchase_date']).dt.total_seconds() / 86400
df['olist_predicted_days'] = (df['estimated_date'] - df['purchase_date']).dt.total_seconds() / 86400
df = df[(df['actual_days'] > 0) & (df['actual_days'] < 60)]
df = df[(df['olist_predicted_days'] > 0) & (df['olist_predicted_days'] < 80)]

# 피처
from sklearn.preprocessing import LabelEncoder
le_cust = LabelEncoder()
le_sell = LabelEncoder()
le_cat = LabelEncoder()

df['purchase_dow'] = df['purchase_date'].dt.dayofweek
df['purchase_hour'] = df['purchase_date'].dt.hour
df['purchase_month'] = df['purchase_date'].dt.month
df['same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
df['product_volume'] = df['product_length_cm'].fillna(0) * df['product_height_cm'].fillna(0) * df['product_width_cm'].fillna(0)
cat_counts = df['category'].value_counts()
df['category_encoded'] = df['category'].map(cat_counts).fillna(0)
cust_counts = df['customer_state'].value_counts()
sell_counts = df['seller_state'].value_counts()
df['customer_state_encoded'] = df['customer_state'].map(cust_counts)
df['seller_state_encoded'] = df['seller_state'].map(sell_counts)
df['customer_state_le'] = le_cust.fit_transform(df['customer_state'].fillna('unknown'))
df['seller_state_le'] = le_sell.fit_transform(df['seller_state'].fillna('unknown'))
df['category_le'] = le_cat.fit_transform(df['category'].fillna('unknown'))
df['route'] = df['seller_state'] + '_' + df['customer_state']

features = [
    'customer_state_le', 'seller_state_le', 'category_le',
    'same_state', 'product_weight_g', 'product_volume', 'product_photos_qty',
    'price', 'freight_value',
    'purchase_dow', 'purchase_hour', 'purchase_month',
    'customer_state_encoded', 'seller_state_encoded', 'category_encoded',
]
df[features] = df[features].fillna(0)

# 시간순 분할
df = df.sort_values('purchase_date').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df[features]
y_train = train_df['actual_days']
X_test = test_df[features]
y_test = test_df['actual_days']

print(f"학습: {len(train_df):,}행, 테스트: {len(test_df):,}행\n")

# ============================================================
# 2. 기본 모델 (복습)
# ============================================================
params_base = {
    'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7,
    'num_leaves': 63, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 42,
}
model_base = lgb.LGBMRegressor(**params_base)
model_base.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.log_evaluation(0)])
test_df['pred_base'] = model_base.predict(X_test)

# ============================================================
# 3. 방법 A: 경로별 맞춤 버퍼
# ============================================================
print("=" * 70)
print("  방법 A: 경로별 맞춤 버퍼")
print("=" * 70)

# 학습 데이터에서 경로별 모델 오차 계산
train_df['pred_base'] = model_base.predict(X_train)
train_df['residual'] = train_df['actual_days'] - train_df['pred_base']  # 양수 = 모델이 과소예측

route_stats = train_df.groupby('route').agg(
    count=('residual', 'size'),
    mean_residual=('residual', 'mean'),
    std_residual=('residual', 'std'),
    pct_late=('residual', lambda x: (x > 0).mean()),  # 과소예측 비율
    p85_residual=('residual', lambda x: np.percentile(x, 85)),  # 85번째 백분위 잔차
).reset_index()

# 버퍼 계산: 경로별로 다르게
# - 과소예측이 자주 일어나는 경로: 더 큰 버퍼
# - 데이터가 적은 경로: 더 큰 버퍼 (불확실성 반영)
# - 과대예측이 많은 경로: 버퍼 없음
route_stats['data_uncertainty'] = np.where(
    route_stats['count'] < 50,
    2.0,  # 데이터 50건 미만: 2일 추가 불확실성
    np.where(route_stats['count'] < 200, 1.0, 0.0)
)

# 최종 버퍼 = max(0, 85번째 백분위 잔차) + 데이터 불확실성
route_stats['buffer'] = np.maximum(0, route_stats['p85_residual']) + route_stats['data_uncertainty']

# 전체 평균 버퍼 (새로운 경로 대비)
global_buffer = np.maximum(0, np.percentile(train_df['residual'], 85)) + 1.0

print(f"\n경로별 버퍼 통계:")
print(f"  버퍼 평균: {route_stats['buffer'].mean():.1f}일")
print(f"  버퍼 범위: {route_stats['buffer'].min():.1f} ~ {route_stats['buffer'].max():.1f}일")
print(f"  글로벌 기본 버퍼: {global_buffer:.1f}일")

# 테스트에 적용
route_buffer_map = dict(zip(route_stats['route'], route_stats['buffer']))
test_df['route_buffer'] = test_df['route'].map(route_buffer_map).fillna(global_buffer)
test_df['pred_route_buffer'] = test_df['pred_base'] + test_df['route_buffer']

# 버퍼가 큰 경로 vs 작은 경로
print("\n--- 버퍼가 큰 경로 TOP 10 ---")
top_buf = route_stats.nlargest(10, 'buffer')[['route', 'count', 'pct_late', 'p85_residual', 'data_uncertainty', 'buffer']]
print(top_buf.to_string(index=False))

print("\n--- 버퍼가 작은 경로 TOP 10 ---")
low_buf = route_stats.nsmallest(10, 'buffer')[['route', 'count', 'pct_late', 'p85_residual', 'data_uncertainty', 'buffer']]
print(low_buf.to_string(index=False))

# ============================================================
# 4. 방법 B: LightGBM 분위 회귀 (85번째 백분위)
# ============================================================
print("\n" + "=" * 70)
print("  방법 B: LightGBM 분위 회귀 (85th percentile)")
print("=" * 70)

params_q85 = {
    'objective': 'quantile', 'alpha': 0.85,
    'metric': 'quantile', 'verbosity': -1,
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7,
    'num_leaves': 63, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 42,
}
model_q85 = lgb.LGBMRegressor(**params_q85)
model_q85.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.log_evaluation(0)])
test_df['pred_q85'] = model_q85.predict(X_test)

# ============================================================
# 5. 방법 C: 분위 회귀 + 경로별 미세 보정
# ============================================================
print("\n" + "=" * 70)
print("  방법 C: 분위 회귀 + 경로별 보정 (하이브리드)")
print("=" * 70)

# 학습 데이터에서 분위 모델의 잔차 계산
train_df['pred_q85'] = model_q85.predict(X_train)
train_df['residual_q85'] = train_df['actual_days'] - train_df['pred_q85']

route_stats_q85 = train_df.groupby('route').agg(
    count=('residual_q85', 'size'),
    pct_late_q85=('residual_q85', lambda x: (x > 0).mean()),
).reset_index()

# 분위 회귀에서도 여전히 과소예측이 많은 경로에만 추가 버퍼
route_stats_q85['extra_buffer'] = np.where(
    (route_stats_q85['pct_late_q85'] > 0.25) | (route_stats_q85['count'] < 30),
    2.0, 0.0
)

extra_buf_map = dict(zip(route_stats_q85['route'], route_stats_q85['extra_buffer']))
test_df['extra_buffer'] = test_df['route'].map(extra_buf_map).fillna(2.0)
test_df['pred_hybrid'] = test_df['pred_q85'] + test_df['extra_buffer']


# ============================================================
# 6. 전체 비교
# ============================================================
print("\n" + "=" * 70)
print("  최종 비교: 4가지 방법")
print("=" * 70)

methods = {
    'Olist 현재': 'olist_predicted_days',
    'LightGBM 기본': 'pred_base',
    'A. 경로별 버퍼': 'pred_route_buffer',
    'B. 분위회귀 85%': 'pred_q85',
    'C. 하이브리드': 'pred_hybrid',
}

print(f"\n{'방법':<22} {'MAE(일)':>10} {'지연율':>10} {'과대예측률':>12} {'15일+과대':>10}")
print("-" * 70)
for name, col in methods.items():
    mae = mean_absolute_error(test_df['actual_days'], test_df[col])
    late = (test_df['actual_days'] > test_df[col]).mean() * 100
    over = (test_df[col] > test_df['actual_days']).mean() * 100
    big_over = ((test_df[col] - test_df['actual_days']) > 15).mean() * 100
    print(f"{name:<22} {mae:>10.2f} {late:>9.1f}% {over:>11.1f}% {big_over:>9.1f}%")

# 구간별 비교 (최적 방법)
print("\n\n--- 구간별 비교: Olist vs 하이브리드 ---")
test_df['actual_bucket'] = pd.cut(test_df['actual_days'], bins=[0, 7, 14, 21, 30, 60], labels=['~7일', '7~14일', '14~21일', '21~30일', '30일+'])
for bucket in test_df['actual_bucket'].cat.categories:
    mask = test_df['actual_bucket'] == bucket
    if mask.sum() < 10:
        continue
    o_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'olist_predicted_days'])
    h_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'pred_hybrid'])
    o_late = (test_df.loc[mask, 'actual_days'] > test_df.loc[mask, 'olist_predicted_days']).mean() * 100
    h_late = (test_df.loc[mask, 'actual_days'] > test_df.loc[mask, 'pred_hybrid']).mean() * 100
    print(f"  {bucket:>8}: Olist MAE={o_mae:.1f}일(지연{o_late:.0f}%), 하이브리드 MAE={h_mae:.1f}일(지연{h_late:.0f}%) [{mask.sum():,}건]")

# 지역별 비교
print("\n\n--- 지역별: Olist vs 하이브리드 ---")
region_results = []
for state in sorted(test_df['customer_state'].unique()):
    mask = test_df['customer_state'] == state
    if mask.sum() < 50:
        continue
    o_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'olist_predicted_days'])
    h_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'pred_hybrid'])
    o_late = (test_df.loc[mask, 'actual_days'] > test_df.loc[mask, 'olist_predicted_days']).mean() * 100
    h_late = (test_df.loc[mask, 'actual_days'] > test_df.loc[mask, 'pred_hybrid']).mean() * 100
    region_results.append({'state': state, 'n': mask.sum(), 
                          'olist_mae': o_mae, 'hybrid_mae': h_mae,
                          'olist_late': o_late, 'hybrid_late': h_late})

rdf = pd.DataFrame(region_results).sort_values('hybrid_late', ascending=False)
print(f"{'주':>4} {'건수':>6} {'Olist MAE':>10} {'H MAE':>8} {'Olist지연':>10} {'H지연':>8}")
print("-" * 52)
for _, r in rdf.iterrows():
    print(f"{r['state']:>4} {r['n']:>6} {r['olist_mae']:>9.1f}일 {r['hybrid_mae']:>7.1f}일 {r['olist_late']:>9.1f}% {r['hybrid_late']:>7.1f}%")

print("\n\n--- 분석 완료 ---")
