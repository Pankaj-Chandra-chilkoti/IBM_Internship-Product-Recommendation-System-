-- 1. Create the database
CREATE DATABASE ecommerce_db;
-- In MySQL Workbench or CLI:

-- 2. Use the newly created database
USE ecommerce_db;

-- 3. Create the 'signup' table (main table for registration/login)
CREATE TABLE IF NOT EXISTS signup (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. (Optional) Create the 'signin' table
-- This table is NOT necessary if you're already using 'signup' for authentication.
-- Keep it only if you need to track signin logs or credentials separately.
CREATE TABLE IF NOT EXISTS signin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    password VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cart (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    product_name VARCHAR(255),
    image_url TEXT,
    price DECIMAL(10,2),
    quantity INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE signup ADD COLUMN profile_image VARCHAR(255) DEFAULT 'default_profile.png';

DESCRIBE signup;
