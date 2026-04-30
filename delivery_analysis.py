"""
배송 지연 분석: 7단계 순차 분석
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sqlite3
import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

DB_PATH = Path(__file__).parent / "ecommerce.db"
conn = sqlite3.connect(DB_PATH)

def section(num, title):
    print(f"\n{'='*80}")
    print(f"  STEP {num}: {title}")
    print(f"{'='*80}\n")

# ============================================================
# STEP 1: 배송 차이 분포
# ============================================================
section(1, "배송 차이(일) 분포 - 예상 vs 실제")

df1 = pd.read_sql("""
    SELECT 
        ROUND(julianday(order_delivered_customer_date) - julianday(order_estimated_delivery_date)) as delay_days,
        COUNT(*) as cnt
    FROM orders
    WHERE order_delivered_customer_date IS NOT NULL
      AND order_estimated_delivery_date IS NOT NULL
    GROUP BY delay_days
    ORDER BY delay_days
""", conn)

print("전체 배달 완료 주문 수:", df1['cnt'].sum())
print(f"평균 차이: {(df1['delay_days'] * df1['cnt']).sum() / df1['cnt'].sum():.1f}일 (음수=일찍, 양수=늦음)")
print()

# 구간별 요약
df1_summary = pd.read_sql("""
    SELECT 
        CASE 
            WHEN delay_days <= -15 THEN 'A. 15일+ 일찍'
            WHEN delay_days <= -7  THEN 'B. 7~14일 일찍'
            WHEN delay_days <= -1  THEN 'C. 1~6일 일찍'
            WHEN delay_days <= 1   THEN 'D. 정시 (+-1일)'
            WHEN delay_days <= 7   THEN 'E. 1~7일 늦음'
            WHEN delay_days <= 14  THEN 'F. 7~14일 늦음'
            ELSE                        'G. 15일+ 늦음'
        END as delay_bucket,
        COUNT(*) as cnt,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE order_delivered_customer_date IS NOT NULL), 1) as pct
    FROM (
        SELECT ROUND(julianday(order_delivered_customer_date) - julianday(order_estimated_delivery_date)) as delay_days
        FROM orders
        WHERE order_delivered_customer_date IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
    )
    GROUP BY delay_bucket
    ORDER BY delay_bucket
""", conn)
print(df1_summary.to_string(index=False))


# ============================================================
# STEP 2: 배송 차이 구간별 리뷰 점수
# ============================================================
section(2, "배송 차이 구간별 리뷰 점수")

df2 = pd.read_sql("""
    SELECT 
        CASE 
            WHEN delay_days <= -15 THEN 'A. 15일+ 일찍'
            WHEN delay_days <= -7  THEN 'B. 7~14일 일찍'
            WHEN delay_days <= -1  THEN 'C. 1~6일 일찍'
            WHEN delay_days <= 1   THEN 'D. 정시 (+-1일)'
            WHEN delay_days <= 7   THEN 'E. 1~7일 늦음'
            WHEN delay_days <= 14  THEN 'F. 7~14일 늦음'
            ELSE                        'G. 15일+ 늦음'
        END as delay_bucket,
        COUNT(*) as reviews,
        ROUND(AVG(review_score), 2) as avg_score,
        ROUND(SUM(CASE WHEN review_score = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_1star,
        ROUND(SUM(CASE WHEN review_score = 5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_5star
    FROM (
        SELECT 
            ROUND(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)) as delay_days,
            r.review_score
        FROM orders o
        JOIN order_reviews r ON o.order_id = r.order_id
        WHERE o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    )
    GROUP BY delay_bucket
    ORDER BY delay_bucket
""", conn)
print(df2.to_string(index=False))


# ============================================================
# STEP 3: 지연 시 취소율
# ============================================================
section(3, "주문 상태 분포 + 지연과 취소의 관계")

df3a = pd.read_sql("""
    SELECT order_status, COUNT(*) as cnt,
           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 1) as pct
    FROM orders
    GROUP BY order_status
    ORDER BY cnt DESC
""", conn)
print("전체 주문 상태 분포:")
print(df3a.to_string(index=False))

# 취소된 주문은 배달 안 됐으니 delay를 못 구함. 대신 예상 배송일 자체가 긴 주문이 취소가 많은지 확인
print("\n--- 예상 배송 소요일별 취소율 ---")
df3b = pd.read_sql("""
    SELECT 
        CASE
            WHEN est_days <= 7  THEN 'A. ~7일'
            WHEN est_days <= 14 THEN 'B. 8~14일'
            WHEN est_days <= 21 THEN 'C. 15~21일'
            WHEN est_days <= 30 THEN 'D. 22~30일'
            ELSE                     'E. 31일+'
        END as estimated_delivery_bucket,
        COUNT(*) as total_orders,
        SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) as canceled,
        ROUND(SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as cancel_rate_pct
    FROM (
        SELECT 
            order_status,
            ROUND(julianday(order_estimated_delivery_date) - julianday(order_purchase_timestamp)) as est_days
        FROM orders
        WHERE order_estimated_delivery_date IS NOT NULL
    )
    GROUP BY estimated_delivery_bucket
    ORDER BY estimated_delivery_bucket
""", conn)
print(df3b.to_string(index=False))


# ============================================================
# STEP 4: 늦은 건 판매자 탓? 물류 탓?
# ============================================================
section(4, "지연 원인 분리: 판매자(출고) vs 물류(배송)")

df4 = pd.read_sql("""
    SELECT 
        CASE 
            WHEN delay_days <= -7  THEN 'A. 7일+ 일찍'
            WHEN delay_days <= -1  THEN 'B. 1~6일 일찍'
            WHEN delay_days <= 1   THEN 'C. 정시 (+-1일)'
            WHEN delay_days <= 7   THEN 'D. 1~7일 늦음'
            ELSE                        'E. 7일+ 늦음'
        END as delay_bucket,
        COUNT(*) as cnt,
        ROUND(AVG(seller_days), 1) as avg_seller_days,
        ROUND(AVG(transit_days), 1) as avg_transit_days,
        ROUND(AVG(total_days), 1) as avg_total_days
    FROM (
        SELECT 
            ROUND(julianday(order_delivered_customer_date) - julianday(order_estimated_delivery_date)) as delay_days,
            julianday(order_delivered_carrier_date) - julianday(order_purchase_timestamp) as seller_days,
            julianday(order_delivered_customer_date) - julianday(order_delivered_carrier_date) as transit_days,
            julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp) as total_days
        FROM orders
        WHERE order_delivered_customer_date IS NOT NULL
          AND order_delivered_carrier_date IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
    )
    GROUP BY delay_bucket
    ORDER BY delay_bucket
""", conn)
print("avg_seller_days = 주문~택배사 인수 (판매자 책임)")
print("avg_transit_days = 택배사 인수~고객 도착 (물류 책임)")
print("avg_total_days = 주문~도착 (전체)")
print()
print(df4.to_string(index=False))


# ============================================================
# STEP 5: 지역별 지연 패턴
# ============================================================
section(5, "지역별 배송 지연 패턴")

# 고객 주(state)별
print("--- 고객 주(state)별 평균 지연 (상위/하위 5개) ---")
df5a = pd.read_sql("""
    SELECT 
        c.customer_state,
        COUNT(*) as orders,
        ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)), 1) as avg_delay,
        ROUND(SUM(CASE WHEN julianday(o.order_delivered_customer_date) > julianday(o.order_estimated_delivery_date) THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as late_pct
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY c.customer_state
    HAVING orders >= 100
    ORDER BY avg_delay DESC
""", conn)
print(df5a.to_string(index=False))

# 판매자 주 → 고객 주 조합 (가장 늦은 경로)
print("\n--- 가장 늦은 판매자주->고객주 경로 TOP 10 ---")
df5b = pd.read_sql("""
    SELECT 
        s.seller_state as from_state,
        c.customer_state as to_state,
        COUNT(*) as orders,
        ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)), 1) as avg_delay,
        ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)), 1) as avg_total_days
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY s.seller_state, c.customer_state
    HAVING orders >= 50
    ORDER BY avg_delay DESC
    LIMIT 10
""", conn)
print(df5b.to_string(index=False))


# ============================================================
# STEP 6: 카테고리/무게별 지연 패턴
# ============================================================
section(6, "카테고리/무게별 배송 지연")

# 카테고리별 (가장 늦는 카테고리)
print("--- 평균 지연이 큰 카테고리 TOP 10 ---")
df6a = pd.read_sql("""
    SELECT 
        COALESCE(ct.product_category_name_english, p.product_category_name) as category,
        COUNT(*) as orders,
        ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)), 1) as avg_delay,
        ROUND(AVG(p.product_weight_g), 0) as avg_weight_g,
        ROUND(AVG(r.review_score), 2) as avg_review
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY category
    HAVING orders >= 50
    ORDER BY avg_delay DESC
    LIMIT 10
""", conn)
print(df6a.to_string(index=False))

# 무게 구간별
print("\n--- 무게 구간별 평균 지연 ---")
df6b = pd.read_sql("""
    SELECT 
        CASE
            WHEN p.product_weight_g <= 500   THEN 'A. ~500g'
            WHEN p.product_weight_g <= 2000  THEN 'B. 500g~2kg'
            WHEN p.product_weight_g <= 5000  THEN 'C. 2~5kg'
            WHEN p.product_weight_g <= 10000 THEN 'D. 5~10kg'
            ELSE                                  'E. 10kg+'
        END as weight_bucket,
        COUNT(*) as orders,
        ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)), 1) as avg_delay,
        ROUND(SUM(CASE WHEN julianday(o.order_delivered_customer_date) > julianday(o.order_estimated_delivery_date) THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as late_pct
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      AND p.product_weight_g IS NOT NULL
    GROUP BY weight_bucket
    ORDER BY weight_bucket
""", conn)
print(df6b.to_string(index=False))


# ============================================================
# STEP 7: 상습 지연 판매자
# ============================================================
section(7, "상습 지연 판매자 TOP 10")

df7 = pd.read_sql("""
    SELECT 
        s.seller_id,
        s.seller_city,
        s.seller_state,
        COUNT(*) as orders,
        ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)), 1) as avg_delay,
        ROUND(SUM(CASE WHEN julianday(o.order_delivered_customer_date) > julianday(o.order_estimated_delivery_date) THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as late_pct,
        ROUND(AVG(julianday(o.order_delivered_carrier_date) - julianday(o.order_purchase_timestamp)), 1) as avg_seller_days,
        ROUND(AVG(r.review_score), 2) as avg_review
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      AND o.order_delivered_carrier_date IS NOT NULL
    GROUP BY s.seller_id
    HAVING orders >= 30
    ORDER BY avg_delay DESC
    LIMIT 10
""", conn)
print(df7.to_string(index=False))

conn.close()
print("\n\n--- 분석 완료 ---")
