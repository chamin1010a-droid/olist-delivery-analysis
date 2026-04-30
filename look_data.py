"""각 테이블의 실제 데이터 샘플을 확인"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "ecommerce.db"
conn = sqlite3.connect(DB_PATH)

def show(title, sql, n=3):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    df = pd.read_sql_query(sql, conn)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)
    print(df.head(n).to_string(index=False))
    print(f"  (total: {len(df):,} rows)")

# 1. 고객: 누가 샀는지
show("CUSTOMERS - 누가 샀는가", """
    SELECT customer_id, customer_unique_id, customer_city, customer_state
    FROM customers LIMIT 5
""", 5)

# 2. 주문: 언제 주문했는지
show("ORDERS - 언제 주문했는가", """
    SELECT order_id, customer_id, order_status, 
           order_purchase_timestamp, 
           order_delivered_customer_date,
           order_estimated_delivery_date
    FROM orders 
    WHERE order_status = 'delivered'
    LIMIT 5
""", 5)

# 3. 주문 상품: 뭘 샀는지 + 얼마에
show("ORDER_ITEMS - 뭘 얼마에 샀는가", """
    SELECT order_id, order_item_id, product_id, seller_id,
           price, freight_value
    FROM order_items LIMIT 5
""", 5)

# 4. 상품: 어떤 상품인지
show("PRODUCTS - 상품 정보", """
    SELECT product_id, product_category_name, 
           product_weight_g, product_length_cm
    FROM products LIMIT 5
""", 5)

# 5. 카테고리 번역
show("CATEGORY_TRANSLATION - 카테고리 영어 번역", """
    SELECT * FROM category_translation LIMIT 10
""", 10)

# 6. 결제: 어떻게 결제했는지
show("ORDER_PAYMENTS - 어떻게 결제했는가", """
    SELECT order_id, payment_type, payment_installments, payment_value
    FROM order_payments LIMIT 5
""", 5)

# 7. 리뷰: 몇점 줬는지
show("ORDER_REVIEWS - 리뷰", """
    SELECT review_id, order_id, review_score, 
           SUBSTR(review_comment_message, 1, 60) as comment_preview,
           review_creation_date
    FROM order_reviews 
    WHERE review_comment_message IS NOT NULL
    LIMIT 5
""", 5)

# 8. 판매자
show("SELLERS - 판매자 정보", """
    SELECT seller_id, seller_city, seller_state
    FROM sellers LIMIT 5
""", 5)

# === 핵심: 한 주문의 전체 흐름을 추적 ===
print("\n" + "="*70)
print("  FULL STORY: 주문 하나의 전체 흐름 추적")
print("="*70)

# 배송 완료된 주문 중 리뷰가 있는 것 하나 선택
sample = pd.read_sql_query("""
    SELECT o.order_id
    FROM orders o
    JOIN order_reviews r ON o.order_id = r.order_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
      AND r.review_comment_message IS NOT NULL
    LIMIT 1
""", conn)

oid = sample.iloc[0]['order_id']

print(f"\n  Tracking order: {oid[:20]}...")

# 이 주문의 고객
print("\n[1] Customer:")
df = pd.read_sql_query(f"""
    SELECT c.customer_city, c.customer_state
    FROM customers c JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_id = '{oid}'
""", conn)
print(f"    City: {df.iloc[0]['customer_city']}, State: {df.iloc[0]['customer_state']}")

# 이 주문의 상품
print("\n[2] Items ordered:")
df = pd.read_sql_query(f"""
    SELECT oi.price, oi.freight_value,
           ct.product_category_name_english as category
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
    WHERE oi.order_id = '{oid}'
""", conn)
for _, row in df.iterrows():
    print(f"    {row['category']} - price: R${row['price']}, shipping: R${row['freight_value']}")

# 이 주문의 결제
print("\n[3] Payment:")
df = pd.read_sql_query(f"""
    SELECT payment_type, payment_installments, payment_value
    FROM order_payments WHERE order_id = '{oid}'
""", conn)
for _, row in df.iterrows():
    print(f"    {row['payment_type']}, {row['payment_installments']} installments, R${row['payment_value']}")

# 이 주문의 배송
print("\n[4] Delivery:")
df = pd.read_sql_query(f"""
    SELECT order_purchase_timestamp as ordered,
           order_delivered_customer_date as delivered,
           order_estimated_delivery_date as estimated
    FROM orders WHERE order_id = '{oid}'
""", conn)
print(f"    Ordered:   {df.iloc[0]['ordered']}")
print(f"    Delivered: {df.iloc[0]['delivered']}")
print(f"    Estimated: {df.iloc[0]['estimated']}")

# 이 주문의 리뷰
print("\n[5] Review:")
df = pd.read_sql_query(f"""
    SELECT review_score, review_comment_message
    FROM order_reviews WHERE order_id = '{oid}'
""", conn)
print(f"    Score: {df.iloc[0]['review_score']}/5")
print(f"    Comment: {df.iloc[0]['review_comment_message'][:100]}")

conn.close()
