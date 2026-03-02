PRAGMA foreign_keys = ON;

CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  city TEXT NOT NULL
);

CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  price REAL NOT NULL
);

CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  product_id INTEGER NOT NULL,
  order_date TEXT NOT NULL,
  quantity INTEGER NOT NULL,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);

INSERT INTO customers (customer_id, name, city) VALUES
  (1, 'Ava Carter', 'Seattle'),
  (2, 'Ben Ortiz', 'Portland'),
  (3, 'Chloe Kim', 'Seattle'),
  (4, 'Diego Ruiz', 'Austin'),
  (5, 'Elena Patel', 'Chicago'),
  (6, 'Finn Brooks', 'Austin');

INSERT INTO products (product_id, name, price) VALUES
  (1, 'Notebook', 4.50),
  (2, 'Pen', 1.25),
  (3, 'Backpack', 29.99),
  (4, 'Mug', 8.00),
  (5, 'Sticker', 0.99),
  (6, 'Mouse', 19.50);

INSERT INTO orders (order_id, customer_id, product_id, order_date, quantity) VALUES
  (1, 1, 3, '2025-01-05', 1),
  (2, 1, 2, '2025-01-10', 3),
  (3, 2, 1, '2025-01-12', 2),
  (4, 2, 4, '2025-02-02', 1),
  (5, 3, 5, '2025-02-10', 10),
  (6, 3, 2, '2025-02-11', 1),
  (7, 4, 6, '2025-02-20', 1),
  (8, 5, 3, '2025-03-01', 1),
  (9, 5, 1, '2025-03-03', 4),
  (10, 6, 4, '2025-03-05', 2);

