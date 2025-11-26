-- Sample E-commerce Database Schema
-- Use this with database_validator.py for testing

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    registration_date TIMESTAMP NOT NULL
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    weight_kg DECIMAL(8,3),
    category VARCHAR(100)
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY NOT NULL,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(12,2) NOT NULL,
    tax_amount DECIMAL(8,2),
    shipping_cost DECIMAL(6,2),
    discount_percent DECIMAL(5,2) DEFAULT 0.00
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY NOT NULL,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10,2) NOT NULL,
    line_total DECIMAL(12,2) NOT NULL
);

CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    rating DECIMAL(2,1) CHECK (rating >= 1.0 AND rating <= 5.0),
    review_text TEXT,
    review_date TIMESTAMP NOT NULL
);