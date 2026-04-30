"""
Olist E-Commerce DB 탐색 스크립트

이 스크립트는 SQL 쿼리 예제들을 실행하고 결과를 보여준다.
SQL의 핵심 개념을 단계별로 체험할 수 있도록 구성.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "ecommerce.db"

def run_query(conn, title, sql):
    """SQL 쿼리 실행 후 결과를 DataFrame으로 출력"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"SQL:\n{sql.strip()}\n")
    
    df = pd.read_sql_query(sql, conn)
    print(df.to_string(index=False))
    print(f"\n({len(df)} rows)")
    return df


def main():
    conn = sqlite3.connect(DB_PATH)
    
    # ──────────────────────────────────────────────
    # LEVEL 1: SELECT + WHERE (기본 조회)
    # "DB에서 원하는 데이터만 꺼내기"
    # ──────────────────────────────────────────────
    
    run_query(conn, "LEVEL 1-1: DB에 어떤 테이블이 있는지 확인", """
        SELECT name, type 
        FROM sqlite_master 
        WHERE type = 'table'
        ORDER BY name
    """)
    
    run_query(conn, "LEVEL 1-2: 주문 상태별 건수", """
        SELECT order_status, COUNT(*) as count
        FROM orders
        GROUP BY order_status
        ORDER BY count DESC
    """)
    
    run_query(conn, "LEVEL 1-3: 결제 수단별 사용 비율", """
        SELECT 
            payment_type,
            COUNT(*) as count,
            ROUND(AVG(payment_value), 2) as avg_value
        FROM order_payments
        GROUP BY payment_type
        ORDER BY count DESC
    """)
    
    # ──────────────────────────────────────────────
    # LEVEL 2: JOIN (여러 테이블 연결)
    # "이게 SQL의 핵심. pandas merge와 비교해보자"
    # ──────────────────────────────────────────────
    
    run_query(conn, "LEVEL 2-1: 카테고리별 매출 TOP 10 (3개 테이블 JOIN)", """
        SELECT 
            ct.product_category_name_english as category,
            COUNT(*) as total_orders,
            ROUND(SUM(oi.price), 2) as total_revenue
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        GROUP BY ct.product_category_name_english
        ORDER BY total_revenue DESC
        LIMIT 10
    """)
    
    run_query(conn, "LEVEL 2-2: 리뷰 점수가 낮은 카테고리 TOP 10 (4개 테이블 JOIN)", """
        SELECT 
            ct.product_category_name_english as category,
            COUNT(*) as review_count,
            ROUND(AVG(r.review_score), 2) as avg_score
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        GROUP BY ct.product_category_name_english
        HAVING review_count >= 50
        ORDER BY avg_score ASC
        LIMIT 10
    """)
    
    # ──────────────────────────────────────────────
    # LEVEL 3: 비즈니스 질문에 답하기
    # "SQL로 실제 의사결정에 쓸 인사이트 뽑기"
    # ──────────────────────────────────────────────
    
    run_query(conn, "LEVEL 3-1: 배송 지연과 리뷰 점수의 관계", """
        SELECT 
            CASE 
                WHEN julianday(o.order_delivered_customer_date) > julianday(o.order_estimated_delivery_date) 
                THEN 'late'
                ELSE 'on_time'
            END as delivery_status,
            COUNT(*) as order_count,
            ROUND(AVG(r.review_score), 2) as avg_review_score
        FROM orders o
        JOIN order_reviews r ON o.order_id = r.order_id
        WHERE o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
        GROUP BY delivery_status
    """)
    
    run_query(conn, "LEVEL 3-2: 월별 매출 추이 (시계열)", """
        SELECT 
            SUBSTR(o.order_purchase_timestamp, 1, 7) as month,
            COUNT(DISTINCT o.order_id) as orders,
            ROUND(SUM(oi.price), 0) as revenue
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY month
        ORDER BY month
    """)
    
    run_query(conn, "LEVEL 3-3: 지역별 주문 수 TOP 10 (고객 위치 기준)", """
        SELECT 
            c.customer_state as state,
            COUNT(*) as total_orders,
            ROUND(SUM(p.payment_value), 0) as total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_payments p ON o.order_id = p.order_id
        GROUP BY c.customer_state
        ORDER BY total_orders DESC
        LIMIT 10
    """)
    
    conn.close()
    print("\n" + "="*60)
    print("  Done! Try modifying these queries or writing your own.")
    print("="*60)


if __name__ == "__main__":
    main()
