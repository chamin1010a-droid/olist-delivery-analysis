"""
배송 소요일 예측 모델: Olist 현재 예측 vs LightGBM

목표: 주문 시점에 알 수 있는 정보만으로 실제 배송 소요일을 예측하여,
Olist의 현재 예상 도착일보다 정확한 예측이 가능함을 보인다.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / "ecommerce.db"
conn = sqlite3.connect(DB_PATH)

# ============================================================
# 1. 데이터 준비: 주문 시점에 알 수 있는 피처만 사용
# ============================================================
print("=" * 70)
print("  STEP 1: 데이터 준비")
print("=" * 70)

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

print(f"원본 데이터: {len(df):,}행")

# 날짜 파싱
df['purchase_date'] = pd.to_datetime(df['order_purchase_timestamp'])
df['estimated_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['delivered_date'] = pd.to_datetime(df['order_delivered_customer_date'])

# 타겟: 실제 배송 소요일
df['actual_days'] = (df['delivered_date'] - df['purchase_date']).dt.total_seconds() / 86400

# Olist의 현재 예측: 예상 소요일
df['olist_predicted_days'] = (df['estimated_date'] - df['purchase_date']).dt.total_seconds() / 86400

# 이상치 제거 (음수이거나 극단적인 값)
df = df[(df['actual_days'] > 0) & (df['actual_days'] < 60)]
df = df[(df['olist_predicted_days'] > 0) & (df['olist_predicted_days'] < 80)]
print(f"이상치 제거 후: {len(df):,}행")

# ============================================================
# 2. 피처 엔지니어링 (주문 시점에 알 수 있는 것만!)
# ============================================================
print("\n" + "=" * 70)
print("  STEP 2: 피처 엔지니어링")
print("=" * 70)

# 시간 피처
df['purchase_dow'] = df['purchase_date'].dt.dayofweek      # 요일 (0=월~6=일)
df['purchase_hour'] = df['purchase_date'].dt.hour            # 시간대
df['purchase_month'] = df['purchase_date'].dt.month          # 월

# 같은 주 여부
df['same_state'] = (df['customer_state'] == df['seller_state']).astype(int)

# 상품 부피 (대략적)
df['product_volume'] = (
    df['product_length_cm'].fillna(0) * 
    df['product_height_cm'].fillna(0) * 
    df['product_width_cm'].fillna(0)
)

# 카테고리 인코딩 (빈도 기반)
cat_counts = df['category'].value_counts()
df['category_encoded'] = df['category'].map(cat_counts).fillna(0)

# 지역 인코딩 (빈도 기반)
cust_counts = df['customer_state'].value_counts()
sell_counts = df['seller_state'].value_counts()
df['customer_state_encoded'] = df['customer_state'].map(cust_counts)
df['seller_state_encoded'] = df['seller_state'].map(sell_counts)

# Label encoding for states (LightGBM categorical)
from sklearn.preprocessing import LabelEncoder
le_cust = LabelEncoder()
le_sell = LabelEncoder()
le_cat = LabelEncoder()
df['customer_state_le'] = le_cust.fit_transform(df['customer_state'].fillna('unknown'))
df['seller_state_le'] = le_sell.fit_transform(df['seller_state'].fillna('unknown'))
df['category_le'] = le_cat.fit_transform(df['category'].fillna('unknown'))

# 피처 목록
features = [
    'customer_state_le', 'seller_state_le', 'category_le',
    'same_state',
    'product_weight_g', 'product_volume', 'product_photos_qty',
    'price', 'freight_value',
    'purchase_dow', 'purchase_hour', 'purchase_month',
    'customer_state_encoded', 'seller_state_encoded', 'category_encoded',
]

target = 'actual_days'

# 결측 처리
df[features] = df[features].fillna(0)

print(f"피처 수: {len(features)}개")
print(f"피처 목록: {features}")

# ============================================================
# 3. 학습/테스트 분할 (시간 순서 기반)
# ============================================================
print("\n" + "=" * 70)
print("  STEP 3: 모델 학습")
print("=" * 70)

# 시간순 정렬 후 80:20 분할 (미래 예측 시뮬레이션)
df = df.sort_values('purchase_date').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"학습 데이터: {len(train_df):,}행 ({train_df['purchase_date'].min().date()} ~ {train_df['purchase_date'].max().date()})")
print(f"테스트 데이터: {len(test_df):,}행 ({test_df['purchase_date'].min().date()} ~ {test_df['purchase_date'].max().date()})")

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# LightGBM 학습
params = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
}

model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.log_evaluation(0)],  # 조용히
)

# 예측
test_df = test_df.copy()
test_df['lgb_predicted_days'] = model.predict(X_test)

# ============================================================
# 4. 성능 비교: Olist vs LightGBM
# ============================================================
print("\n" + "=" * 70)
print("  STEP 4: 성능 비교")
print("=" * 70)

olist_mae = mean_absolute_error(test_df['actual_days'], test_df['olist_predicted_days'])
lgb_mae = mean_absolute_error(test_df['actual_days'], test_df['lgb_predicted_days'])
olist_rmse = np.sqrt(mean_squared_error(test_df['actual_days'], test_df['olist_predicted_days']))
lgb_rmse = np.sqrt(mean_squared_error(test_df['actual_days'], test_df['lgb_predicted_days']))

# Olist는 항상 과대예측하니까, 과대/과소 비율도 보자
olist_over = (test_df['olist_predicted_days'] > test_df['actual_days']).mean() * 100
lgb_over = (test_df['lgb_predicted_days'] > test_df['actual_days']).mean() * 100

print(f"\n{'지표':<25} {'Olist 현재':>15} {'LightGBM':>15} {'개선':>10}")
print("-" * 70)
print(f"{'MAE (평균 절대 오차, 일)':<25} {olist_mae:>15.2f} {lgb_mae:>15.2f} {(olist_mae-lgb_mae)/olist_mae*100:>9.1f}%")
print(f"{'RMSE (제곱근 평균 오차, 일)':<25} {olist_rmse:>15.2f} {lgb_rmse:>15.2f} {(olist_rmse-lgb_rmse)/olist_rmse*100:>9.1f}%")
print(f"{'과대예측 비율':<25} {olist_over:>14.1f}% {lgb_over:>14.1f}%")

# 지연율 시뮬레이션: 예측일보다 늦게 도착한 비율
olist_late = (test_df['actual_days'] > test_df['olist_predicted_days']).mean() * 100
lgb_late = (test_df['actual_days'] > test_df['lgb_predicted_days']).mean() * 100
print(f"{'지연율 (예측보다 늦은 비율)':<25} {olist_late:>14.1f}% {lgb_late:>14.1f}%")

# 구간별 비교
print("\n--- 구간별 MAE 비교 ---")
test_df['actual_bucket'] = pd.cut(test_df['actual_days'], bins=[0, 7, 14, 21, 30, 60], labels=['~7일', '7~14일', '14~21일', '21~30일', '30일+'])
for bucket in test_df['actual_bucket'].cat.categories:
    mask = test_df['actual_bucket'] == bucket
    if mask.sum() < 10:
        continue
    o_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'olist_predicted_days'])
    l_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'lgb_predicted_days'])
    print(f"  실제 {bucket:>8}: Olist MAE={o_mae:.1f}일, LightGBM MAE={l_mae:.1f}일, 개선 {(o_mae-l_mae)/o_mae*100:.0f}% ({mask.sum():,}건)")

# ============================================================
# 5. Feature Importance
# ============================================================
print("\n" + "=" * 70)
print("  STEP 5: 중요 피처 TOP 10")
print("=" * 70)

importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string(index=False))

# ============================================================
# 6. 지역별 개선 효과
# ============================================================
print("\n" + "=" * 70)
print("  STEP 6: 지역별 개선 효과")
print("=" * 70)

region_results = []
for state in test_df['customer_state'].unique():
    mask = test_df['customer_state'] == state
    if mask.sum() < 50:
        continue
    o_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'olist_predicted_days'])
    l_mae = mean_absolute_error(test_df.loc[mask, 'actual_days'], test_df.loc[mask, 'lgb_predicted_days'])
    region_results.append({
        'state': state,
        'orders': mask.sum(),
        'olist_mae': round(o_mae, 1),
        'lgb_mae': round(l_mae, 1),
        'improvement_pct': round((o_mae - l_mae) / o_mae * 100, 1)
    })

region_df = pd.DataFrame(region_results).sort_values('improvement_pct', ascending=False)
print(region_df.to_string(index=False))

print("\n\n--- 분석 완료 ---")
