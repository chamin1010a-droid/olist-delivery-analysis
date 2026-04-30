"""
Olist E-Commerce CSV → SQLite 변환 스크립트
CSV 파일들을 SQLite 데이터베이스로 로드하고, 테이블 간 관계(외래키)를 설정한다.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = Path(__file__).parent / "ecommerce.db"

def create_database():
    """CSV 파일들을 SQLite DB로 변환"""
    
    # 기존 DB가 있으면 삭제 후 새로 생성
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    
    # ──────────────────────────────────────
    # 1. 테이블 생성 (관계 포함)
    # ──────────────────────────────────────
    
    conn.executescript("""
    -- 상품 카테고리 번역 (포르투갈어 → 영어)
    CREATE TABLE category_translation (
        product_category_name TEXT PRIMARY KEY,
        product_category_name_english TEXT
    );
    
    -- 고객
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        customer_unique_id TEXT,
        customer_zip_code_prefix TEXT,
        customer_city TEXT,
        customer_state TEXT
    );
    
    -- 판매자
    CREATE TABLE sellers (
        seller_id TEXT PRIMARY KEY,
        seller_zip_code_prefix TEXT,
        seller_city TEXT,
        seller_state TEXT
    );
    
    -- 상품
    CREATE TABLE products (
        product_id TEXT PRIMARY KEY,
        product_category_name TEXT,
        product_name_lenght INTEGER,
        product_description_lenght INTEGER,
        product_photos_qty INTEGER,
        product_weight_g INTEGER,
        product_length_cm INTEGER,
        product_height_cm INTEGER,
        product_width_cm INTEGER
    );
    
    -- 주문
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT,
        order_status TEXT,
        order_purchase_timestamp TEXT,
        order_approved_at TEXT,
        order_delivered_carrier_date TEXT,
        order_delivered_customer_date TEXT,
        order_estimated_delivery_date TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    
    -- 주문 상품 (주문-상품 중간 테이블)
    CREATE TABLE order_items (
        order_id TEXT,
        order_item_id INTEGER,
        product_id TEXT,
        seller_id TEXT,
        shipping_limit_date TEXT,
        price REAL,
        freight_value REAL,
        PRIMARY KEY (order_id, order_item_id),
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id),
        FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
    );
    
    -- 결제
    CREATE TABLE order_payments (
        order_id TEXT,
        payment_sequential INTEGER,
        payment_type TEXT,
        payment_installments INTEGER,
        payment_value REAL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    
    -- 리뷰
    CREATE TABLE order_reviews (
        review_id TEXT,
        order_id TEXT,
        review_score INTEGER,
        review_comment_title TEXT,
        review_comment_message TEXT,
        review_creation_date TEXT,
        review_answer_timestamp TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    
    -- 지역 정보
    CREATE TABLE geolocation (
        geolocation_zip_code_prefix TEXT,
        geolocation_lat REAL,
        geolocation_lng REAL,
        geolocation_city TEXT,
        geolocation_state TEXT
    );
    """)
    
    # ──────────────────────────────────────
    # 2. CSV 데이터 로드
    # ──────────────────────────────────────
    
    file_table_map = {
        "product_category_name_translation.csv": "category_translation",
        "olist_customers_dataset.csv": "customers",
        "olist_sellers_dataset.csv": "sellers",
        "olist_products_dataset.csv": "products",
        "olist_orders_dataset.csv": "orders",
        "olist_order_items_dataset.csv": "order_items",
        "olist_order_payments_dataset.csv": "order_payments",
        "olist_order_reviews_dataset.csv": "order_reviews",
        "olist_geolocation_dataset.csv": "geolocation",
    }
    
    for csv_file, table_name in file_table_map.items():
        csv_path = DATA_DIR / csv_file
        print(f"  로딩: {csv_file} → {table_name}...", end=" ")
        
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="append", index=False)
        
        # 로드된 행 수 확인
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"-> {count:,} rows")
    
    # ──────────────────────────────────────
    # 3. 유용한 인덱스 추가
    # ──────────────────────────────────────
    
    conn.executescript("""
    CREATE INDEX idx_orders_customer ON orders(customer_id);
    CREATE INDEX idx_orders_status ON orders(order_status);
    CREATE INDEX idx_order_items_product ON order_items(product_id);
    CREATE INDEX idx_order_items_seller ON order_items(seller_id);
    CREATE INDEX idx_payments_order ON order_payments(order_id);
    CREATE INDEX idx_reviews_order ON order_reviews(order_id);
    CREATE INDEX idx_geolocation_zip ON geolocation(geolocation_zip_code_prefix);
    """)
    
    conn.commit()
    conn.close()
    
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDB created: {DB_PATH}")
    print(f"Size: {db_size_mb:.1f} MB")


if __name__ == "__main__":
    print("=" * 50)
    print("Olist E-Commerce -> SQLite DB")
    print("=" * 50)
    create_database()
