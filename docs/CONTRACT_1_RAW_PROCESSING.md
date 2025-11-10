# Data Contract 1: Raw Data Processing

## 🔍 **Contract Overview**

**Contract ID**: DC-001
**Version**: 1.0.0
**Owner**: Data Engineering Team
**Status**: Active
**Created**: 2025-11-10
**Review Date**: 2025-11-17

## 📥 **Input Specifications**

### **Primary Data Sources**

| Source | Format | Encoding | Size | Records | Location |
|--------|--------|----------|------|---------|----------|
| `articles_filtered.csv` | CSV | UTF-8 | 4.9MB | 15,581 | `data/raw/` |
| `customers_filtered.csv` | CSV | ASCII | 4.1MB | 28,317 | `data/raw/` |
| `transactions_train_2M.csv` | CSV | ASCII | 11MB | 100,000 | `data/raw/` |

### **Data Quality Requirements**

```python
validation_thresholds:
  null_tolerance: 0.05  # Max 5% null values per column
  duplicate_tolerance: 0.001  # Max 0.1% duplicates
  referential_integrity: 1.0  # 100% foreign key validation
  encoding_validation: true  # Validate file encodings
```

### **Schema Validation**

#### **Articles Schema**
```yaml
required_fields:
  - article_id: string (primary_key)
  - product_code: int64
  - prod_name: string
  - product_type_no: int64
  - product_type_name: string
  - product_group_name: string
  - colour_group_name: string
  - department_name: string
  - index_name: string
  - section_name: string
  - garment_group_name: string
  - detail_desc: string

validation_rules:
  - article_id: "unique, not_null"
  - product_code: "positive_integer"
  - prod_name: "not_empty"
```

#### **Customers Schema**
```yaml
required_fields:
  - customer_id: string (primary_key, hashed)
  - FN: float64 (nullable, binary_indicator)
  - Active: float64 (nullable, binary_indicator)
  - club_member_status: string (enum: ACTIVE, PRE-CREATION, LEFT CLUB)
  - fashion_news_frequency: string (enum: NONE, Regularly, Monthly)
  - age: int64
  - postal_code: string (hashed)

validation_rules:
  - customer_id: "unique, not_null, sha256_hashed"
  - age: "between(16, 99)"
  - club_member_status: "in_valid_enum"
```

#### **Transactions Schema**
```yaml
required_fields:
  - t_dat: datetime64 (YYYY-MM-DD)
  - customer_id: string (foreign_key)
  - article_id: string (foreign_key)
  - price: float64 (normalized 0-1)
  - sales_channel_id: int64 (enum: 1, 2)

validation_rules:
  - t_dat: "between('2018-09-20', '2018-09-22')"
  - price: "between(0, 1)"
  - sales_channel_id: "in([1, 2])"
```

## ⚙️ **Processing Pipeline**

### **Phase 1: Data Cleaning & Validation**
```python
cleaning_operations:
  1. Schema Validation:
     - Validate required columns present
     - Check data types match expected
     - Verify encoding integrity

  2. Quality Checks:
     - Detect and remove duplicates
     - Handle missing values according to business rules
     - Validate referential integrity

  3. Data Consistency:
     - Standardize categorical values
     - Normalize text fields
     - Validate temporal consistency
```

### **Phase 2: Feature Engineering**
```python
feature_engineering:
  temporal_features:
    - day_of_week: "from t_dat"
    - month: "from t_dat"
    - season: "derived from month"
    - days_since_first_purchase: "per customer"
    - purchase_frequency: "per customer"

  user_features:
    - customer_lifetime_days: "max_date - min_date"
    - total_purchases: "count per customer"
    - avg_basket_size: "mean price per transaction"
    - preferred_channel: "mode of sales_channel_id"
    - age_group: "bucketized age (16-24, 25-34, 35-44, 45+)"

  item_features:
    - popularity_score: "log(transaction_count + 1)"
    - price_category: "quartile-based"
    - color_intensity: "derived from color_group"
    - category_hierarchy: "concatenated from product_type to garment_group"

  interaction_features:
    - user_item_price_preference: "deviation from user average"
    - temporal_popularity: "popularity within time windows"
    - channel_affinity: "user preference per channel"
```

### **Phase 3: Statistical Analysis & Profiling**
```python
statistical_analysis:
  distributions:
    - customer_age_distribution: "histogram, stats"
    - price_distribution: "log-scale analysis"
    - temporal_patterns: "time series analysis"
    - category_popularity: "power law analysis"

  data_quality_metrics:
    - completeness_score: "per column"
    - uniqueness_score: "per entity"
    - consistency_score: "across tables"
    - timeliness_score: "data freshness"

  business_insights:
    - top_selling_categories: "by volume and revenue"
    - customer_segments: "clustering analysis"
    - seasonal_patterns: "temporal trends"
    - channel_performance: "online vs offline"
```

## 📤 **Output Specifications**

### **Processed Files**

| File | Format | Compression | Partitioning | Size Estimate |
|------|--------|-------------|--------------|---------------|
| `articles.parquet` | Parquet | Snappy | product_group_name | ~8MB |
| `customers.parquet` | Parquet | Snappy | None | ~6MB |
| `transactions.parquet` | Parquet | Snappy | year/month | ~15MB |
| `user_item_interactions.parquet` | Parquet | Snappy | customer_segment | ~20MB |
| `data_profile.json` | JSON | None | None | ~500KB |

### **Schema Definitions**

#### **Processed Articles Schema**
```python
articles_schema = {
    'article_id': 'string',
    'product_code': 'int32',
    'product_type_name': 'string',
    'product_group_name': 'string',
    'colour_group_name': 'string',
    'department_name': 'string',
    'index_name': 'string',
    'section_name': 'string',
    'garment_group_name': 'string',
    'detail_desc': 'string',
    'popularity_score': 'float32',
    'price_category': 'category',
    'category_hierarchy': 'string'
}
```

#### **Processed Customers Schema**
```python
customers_schema = {
    'customer_id': 'string',
    'FN': 'float32',
    'Active': 'float32',
    'club_member_status': 'category',
    'fashion_news_frequency': 'category',
    'age': 'int16',
    'age_group': 'category',
    'postal_code': 'string',
    'customer_lifetime_days': 'int16',
    'total_purchases': 'int32',
    'avg_basket_size': 'float32',
    'preferred_channel': 'int8'
}
```

#### **Processed Transactions Schema**
```python
transactions_schema = {
    't_dat': 'datetime64[ns]',
    'customer_id': 'string',
    'article_id': 'string',
    'price': 'float32',
    'sales_channel_id': 'int8',
    'day_of_week': 'int8',
    'month': 'int8',
    'season': 'category',
    'user_item_price_preference': 'float32'
}
```

## ✅ **Quality Gates & Validation**

### **Pre-Processing Validation**
```python
pre_processing_checks:
  file_integrity:
    - all_input_files_exist: true
    - file_encodings_valid: true
    - file_headers_correct: true
    - no_corrupted_data: true

  schema_validation:
    - required_columns_present: true
    - data_types_correct: true
    - primary_keys_unique: true
    - foreign_keys_valid: true
```

### **Post-Processing Validation**
```python
post_processing_checks:
  data_quality:
    - no_null_primary_keys: true
    - referential_integrity_maintained: true
    - date_ranges_valid: true
    - categorical_values_valid: true

  business_logic:
    - all_prices_positive: true
    - customer_ages_reasonable: true
    - transaction_dates_sequential: true
    - channel_assignments_correct: true

  performance:
    - processing_time_minutes: "< 10"
    - memory_usage_gb: "< 4"
    - output_size_reasonable: true
    - data_loss_percent: "< 1"
```

### **Error Handling**
```python
error_handling:
  critical_errors:
    - schema_validation_failure: "STOP_PROCESSING"
    - referential_integrity_violation: "STOP_PROCESSING"
    - file_corruption: "STOP_PROCESSING"
    - encoding_issues: "STOP_PROCESSING"

  warnings:
    - high_null_percentage: "LOG_AND_CONTINUE"
    - minor_data_type_mismatches: "AUTO_CONVERT"
    - slight_data_loss: "LOG_AND_CONTINUE"
    - performance_degradation: "LOG_AND_CONTINUE"
```

## 📊 **Performance Requirements**

### **Processing Metrics**
```python
performance_requirements:
  throughput:
    - minimum_processing_rate: "5000 records/second"
    - target_processing_rate: "10000 records/second"

  resource_usage:
    - max_memory_usage: "4GB"
    - max_cpu_usage: "80%"
    - max_disk_io: "100MB/second"

  timing:
    - total_processing_time: "< 600 seconds" (10 minutes)
    - per_table_processing:
      articles: "< 60 seconds"
      customers: "< 90 seconds"
      transactions: "< 300 seconds"
```

### **Scalability Requirements**
```python
scalability:
  local_dataset:
    - records: "100K transactions"
    - memory: "4GB sufficient"
    - processing_time: "< 10 minutes"

  production_dataset:
    - records: "34M+ transactions"
    - memory: "64GB+ required"
    - processing_time: "< 4 hours"
    - distributed_processing: "required"
```

## 📝 **Metadata & Lineage**

### **Data Lineage**
```python
lineage_tracking:
  source_system: "H&M e-commerce platform"
  extraction_date: "2025-11-10"
  processing_version: "1.0.0"
  processing_engine: "Python 3.11 + Polars"

  transformations:
    articles:
      - text_cleaning: "standardize product descriptions"
      - category_standardization: "map to canonical categories"
      - popularity_calculation: "log(transaction_count + 1)"

    customers:
      - age_bucketization: "5-year age groups"
      - engagement_scoring: "based on FN and Active flags"
      - lifetime_calculation: "days between first and last transaction"

    transactions:
      - price_normalization: "min-max scaling to 0-1 range"
      - temporal_features: "day_of_week, month, season"
      - user_item_features: "interaction-specific metrics"
```

### **Quality Metrics**
```python
quality_metrics:
  completeness:
    articles: "99.8%"
    customers: "98.5%" (FN/Active nullable)
    transactions: "100%"

  accuracy:
    referential_integrity: "100%"
    temporal_consistency: "100%"
    categorical_consistency: "100%"

  consistency:
    schema_conformance: "100%"
    data_type_conformance: "100%"
    business_rule_compliance: "100%"
```

## 🚨 **Monitoring & Alerting**

### **Automated Monitoring**
```python
monitoring:
  real_time_metrics:
    - processing_progress: "percentage complete"
    - error_rate: "errors/1000 records"
    - memory_usage: "current MB"
    - processing_rate: "records/second"

  post_processing_alerts:
    - data_quality_degradation: "score < 95%"
    - performance_regression: "time > 15 minutes"
    - data_loss_threshold: "> 2% records lost"
    - schema_drift: "unexpected schema changes"
```

### **Notification Channels**
```python
notifications:
  success:
    - slack_channel: "#data-pipeline-success"
    - email: "data-team@company.com"
    - logging: "INFO level"

  warnings:
    - slack_channel: "#data-pipeline-warnings"
    - email: "data-team@company.com"
    - logging: "WARN level"

  errors:
    - slack_channel: "#data-pipeline-alerts"
    - email: "data-team@company.com, eng-team@company.com"
    - logging: "ERROR level"
    - pagerduty: "critical errors only"
```

## 🔐 **Security & Compliance**

### **Data Privacy**
```python
privacy_measures:
  customer_identification:
    - customer_id: "SHA-256 hashed"
    - postal_code: "SHA-256 hashed"
    - no_pii_in_logs: "true"

  access_control:
    - raw_data_access: "data-team only"
    - processed_data_access: "ml-team, data-team"
    - audit_logging: "all access attempts"
```

### **Data Governance**
```python
governance:
  data_retention:
    - raw_data: "90 days"
    - processed_data: "365 days"
    - logs: "30 days"

  compliance:
    - gdpr_compliant: "true"
    - data_classification: "confidential"
    - encryption_at_rest: "true"
    - encryption_in_transit: "true"
```

---

## 📋 **Implementation Checklist**

### **Pre-Execution**
- [ ] Verify input files exist and are accessible
- [ ] Validate file formats and encodings
- [ ] Check available disk space (minimum 50GB)
- [ ] Validate Python environment and dependencies

### **During Execution**
- [ ] Monitor processing progress and performance
- [ ] Validate intermediate results
- [ ] Check memory and resource usage
- [ ] Log all transformations and decisions

### **Post-Execution**
- [ ] Verify all output files created successfully
- [ ] Validate output schemas and data types
- [ ] Run quality gates and validation checks
- [ ] Generate processing report and metrics
- [ ] Archive processing logs and metadata

### **Documentation**
- [ ] Update data catalog with new dataset information
- [ ] Document any data quality issues discovered
- [ ] Create data lineage documentation
- [ ] Archive processing parameters and configuration

---

**Contract Status**: ✅ Active
**Next Review**: 2025-11-17
**Implementation Owner**: Data Engineering Team
**Stakeholders**: ML Engineering Team, Data Science Team, Business Analytics