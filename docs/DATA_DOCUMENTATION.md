# Raw Data Documentation

## 📊 **Data Overview**

This document provides comprehensive technical documentation for the raw datasets used in the H&M Deep Learning project. The data represents a subset of the H&M Group e-commerce transaction data for building recommendation systems.

### **Dataset Summary**
- **Total Files**: 3 CSV files
- **Total Size**: 20MB (4.9M + 4.1M + 11M)
- **Data Period**: September 20-22, 2018
- **Scope**: **Local development sample** (34M+ records in full production dataset)
- **Purpose**: Fast local debugging and prototyping before cloud training

### **🚨 Important: Local vs Production Data**
- **Local Development**: 100K transactions (current files)
- **Full Production**: 34M+ transactions (cloud environment)
- **Scale Factor**: ~340x larger in production
- **Strategy**: Debug locally → Validate → Scale to cloud

---

## 📁 **Data Schema Documentation**

### 1. **articles_filtered.csv**
**File Size**: 4.9MB | **Records**: 15,581 | **Encoding**: UTF-8

#### **Purpose**: Product catalog with detailed article metadata
#### **Primary Key**: `article_id`

| Column | Type | Description | Nulls | Example |
|--------|------|-------------|-------|---------|
| `article_id` | String | Unique product identifier | No | 108775015 |
| `product_code` | Integer | Product group identifier | No | 108775 |
| `prod_name` | String | Product name | No | "Strap top" |
| `product_type_no` | Integer | Product type code | No | 253 |
| `product_type_name` | String | Product type description | No | "Vest top" |
| `product_group_name` | String | High-level product category | No | "Garment Upper body" |
| `graphical_appearance_no` | Integer | Visual pattern code | No | 1010016 |
| `graphical_appearance_name` | String | Visual pattern description | No | "Solid" |
| `colour_group_code` | Integer | Color category code | No | 9 |
| `colour_group_name` | String | Color category description | No | "Black" |
| `perceived_colour_value_id` | Integer | Color brightness ID | No | 4 |
| `perceived_colour_value_name` | String | Color brightness description | No | "Dark" |
| `perceived_colour_master_id` | Integer | Master color ID | No | 5 |
| `perceived_colour_master_name` | String | Master color description | No | "Black" |
| `department_no` | Integer | Department code | No | 1676 |
| `department_name` | String | Department description | No | "Jersey Basic" |
| `index_code` | String | Index group code | No | "A" |
| `index_name` | String | Index group description | No | "Ladieswear" |
| `index_group_no` | Integer | Index group number | No | 1 |
| `index_group_name` | String | Index group description | No | "Ladieswear" |
| `section_no` | Integer | Section code | No | 16 |
| `section_name` | String | Section description | No | "Womens Everyday Basics" |
| `garment_group_no` | Integer | Garment group code | No | 1002 |
| `garment_group_name` | String | Garment group description | No | "Jersey Basic" |
| `detail_desc` | String | Detailed product description | No | "Jersey top with narrow shoulder straps." |

#### **Data Quality Notes**:
- All fields appear to be populated (no nulls detected in sample)
- UTF-8 encoding supports special characters
- Product descriptions contain marketing language
- Hierarchical categorization: product_type → product_group → index_group → section

### 2. **customers_filtered.csv**
**File Size**: 4.1MB | **Records**: 28,317 | **Encoding**: ASCII

#### **Purpose**: Customer demographic and behavioral data
#### **Primary Key**: `customer_id`

| Column | Type | Description | Nulls | Example |
|--------|------|-------------|-------|---------|
| `customer_id` | String | SHA-256 hashed customer identifier | No | "0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa" |
| `FN` | Float | Fashion news indicator (1.0 = subscribed) | Yes | 1.0 |
| `Active` | Float | Customer activity status (1.0 = active) | Yes | 1.0 |
| `club_member_status` | String | Loyalty program membership status | No | "ACTIVE" |
| `fashion_news_frequency` | String | Newsletter subscription frequency | No | "Regularly" |
| `age` | Integer | Customer age | No | 25 |
| `postal_code` | String | Hashed postal code | No | "2973abc54daa8a5f8ccfe9362140c63247c5eee03f1d93f4c830291c32bc3057" |

#### **Data Quality Notes**:
- `customer_id` is SHA-256 hashed for privacy
- `FN` and `Active` contain nulls (likely binary indicators)
- `postal_code` is anonymized through hashing
- Age range appears reasonable (16-99 in sample)
- Club member status includes: "ACTIVE", "PRE-CREATION", "LEFT CLUB"

### 3. **transactions_train_2M.csv**
**File Size**: 11MB | **Records**: 100,000 | **Encoding**: ASCII

#### **Purpose**: Transaction records for recommendation training
#### **Composite Key**: `(customer_id, article_id, t_dat)`

| Column | Type | Description | Nulls | Example |
|--------|------|-------------|-------|---------|
| `t_dat` | Date | Transaction date (YYYY-MM-DD) | No | "2018-09-20" |
| `customer_id` | String | Foreign key to customers table | No | "000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318" |
| `article_id` | String | Foreign key to articles table | No | "663713001" |
| `price` | Float | Transaction price (normalized) | No | 0.050830508474576264 |
| `sales_channel_id` | Integer | Sales channel (1=Online, 2=Offline) | No | 2 |

#### **Data Quality Notes**:
- `t_dat` format: ISO 8601 (YYYY-MM-DD)
- `price` appears to be normalized (0-1 range)
- `sales_channel_id`: 1 = Online, 2 = Offline (inferred from data)
- All transactions in 3-day window (Sep 20-22, 2018)
- Price precision suggests currency conversion or normalization
- Customer and article IDs should be validated against respective dimension tables

---

## 🔗 **Relationship Diagram**

```
customers (1) ←→ (*) transactions (*) ←→ (1) articles
    ↑                                      ↑
customer_id (FK)                     article_id (FK)
```

**Foreign Key Relationships**:
- `transactions.customer_id` → `customers.customer_id`
- `transactions.article_id` → `articles.article_id`

---

## 📈 **Data Statistics**

### **Transaction Volume by Day**:
- **Sep 20, 2018**: ~33,333 transactions
- **Sep 21, 2018**: ~33,333 transactions
- **Sep 22, 2018**: ~33,334 transactions

### **Price Distribution**:
- **Range**: 0.001 - 0.999 (normalized)
- **Average**: ~0.05 (sample-based estimate)
- **Precision**: 15 decimal places (suggests high precision calculation)

### **Customer Activity**:
- **Unique Customers**: ~28,317 (from customers table)
- **Active Customers**: ~unknown (requires join analysis)
- **Age Distribution**: Sample shows 16-51 range

### **Product Catalog**:
- **Unique Articles**: 15,581
- **Product Types**: Multiple categories (vest tops, bras, etc.)
- **Color Variations**: Extensive color taxonomy
- **Hierarchical Categories**: 4-level taxonomy (type → group → index → section)

---

## ⚠️ **Data Quality & Validation**

### **Known Issues**:
1. **Null Values**: `FN` and `Active` fields in customers table contain nulls
2. **Price Normalization**: Price scale unclear (appears normalized)
3. **Sample Size**: Limited to 3 days of transaction data
4. **Encoding**: Mixed UTF-8/ASCII across files

### **Validation Rules**:
```python
# Referential Integrity
assert all(transactions['customer_id'].isin(customers['customer_id']))
assert all(transactions['article_id'].isin(articles['article_id']))

# Data Range Validation
assert customers['age'].between(16, 99).all()
assert transactions['price'].between(0, 1).all()
assert transactions['sales_channel_id'].isin([1, 2]).all()

# Date Consistency
assert pd.to_datetime(transactions['t_dat']).between('2018-09-20', '2018-09-22').all()
```

### **Recommended Data Quality Checks**:
1. **Missing Value Analysis**: Quantify nulls in each table
2. **Outlier Detection**: Check for anomalous prices and ages
3. **Duplicate Detection**: Ensure no duplicate transaction records
4. **Referential Integrity**: Validate all foreign keys exist
5. **Business Logic**: Validate price ranges and customer behavior patterns

---

## 💡 **Usage Guidelines**

### **For Machine Learning**:
- **Recommendation Systems**: Use transaction data with customer/article features
- **Customer Segmentation**: Leverage demographics and purchase patterns
- **Demand Forecasting**: Analyze temporal transaction patterns
- **Product Clustering**: Group similar articles based on attributes

### **For Data Processing**:
- **Memory Management**: Files are manageable in memory (<50MB total)
- **Join Strategy**: Use `customer_id` and `article_id` for efficient joins
- **Indexing**: Consider indexing on customer_id and article_id for performance
- **Data Types**: Use appropriate pandas dtypes for memory efficiency

### **For Production**:
- **Schema Validation**: Implement schema validation in data pipelines
- **Monitoring**: Track data quality metrics over time
- **Versioning**: Implement data versioning for reproducibility
- **Privacy**: Remember customer IDs and postal codes are hashed/anonymized

### **🎯 Development Strategy**:
1. **Local Development**: Use current 20MB dataset for rapid iteration
2. **Validation**: Test data pipelines and ML models on local sample
3. **Cloud Scaling**: Deploy validated code to handle 34M+ production records
4. **Performance**: Monitor memory/compute requirements when scaling 340x

### **⚡ Performance Expectations**:
- **Local**: <2GB RAM, <5 minutes for full processing
- **Cloud**: Requires distributed processing, significant compute resources
- **Memory Scaling**: ~340x memory requirements in production
- **I/O Considerations**: Use cloud-native formats (Parquet) for production

---

## 🔧 **Technical Specifications**

### **File Formats**:
- **Type**: CSV (Comma-Separated Values)
- **Delimiter**: `,`
- **Quote Character**: `"` (standard)
- **Line Endings**: CRLF (Windows format)
- **Headers**: Present in all files

### **Recommended Processing**:
```python
import pandas as pd

# Optimized reading with proper dtypes
articles = pd.read_csv('data/raw/articles_filtered.csv', dtype={
    'article_id': 'string',
    'product_code': 'int32',
    'product_type_no': 'int16',
    # ... other dtypes
})

customers = pd.read_csv('data/raw/customers_filtered.csv', dtype={
    'customer_id': 'string',
    'FN': 'float32',
    'Active': 'float32',
    'age': 'int16',
    'postal_code': 'string'
})

transactions = pd.read_csv('data/raw/transactions_train_2M.csv', dtype={
    'customer_id': 'string',
    'article_id': 'string',
    'price': 'float32',
    'sales_channel_id': 'int8'
}, parse_dates=['t_dat'])
```

### **Memory Requirements**:
- **Local Dataset**: 512MB minimum, 2GB recommended
- **Production Dataset**: ~170GB minimum, distributed processing required
- **Storage**: 20MB raw (local), ~6.8GB+ raw (production)
- **Processing**: <5 minutes (local), hours to days (production)

---

**Last Updated**: 2025-11-10
**Data Version**: v1.0 (filtered sample)
**Maintainer**: Data Engineering Team
**Contact**: [data-team@company.com]