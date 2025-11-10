# Data Contract 2: Training Dataset Generation

## 🔍 **Contract Overview**

**Contract ID**: DC-002
**Version**: 1.0.0
**Owner**: ML Engineering Team
**Status**: Active
**Created**: 2025-11-10
**Review Date**: 2025-11-17
**Dependencies**: DC-001 (Raw Data Processing)

## 📥 **Input Specifications**

### **Primary Data Sources**

| Source | Format | Location | Size | Validation |
|--------|--------|----------|------|------------|
| `articles.parquet` | Parquet | `data/processed/` | ~8MB | Schema validated |
| `customers.parquet` | Parquet | `data/processed/` | ~6MB | Schema validated |
| `transactions.parquet` | Parquet | `data/processed/` | ~15MB | Schema validated |

### **Input Quality Requirements**
```python
input_validation:
  referential_integrity: "100%"
  schema_validation: "passed"
  data_quality_score: "> 95%"
  no_null_primary_keys: "true"
  temporal_consistency: "validated"
```

## 🎯 **Training Dataset Requirements**

### **Core Objectives**
```python
objectives:
  primary: "Generate high-quality contrastive learning pairs"
  secondary: "Enable hard negative mining for model robustness"
  tertiary: "Maintain category balance and temporal consistency"

success_metrics:
  recall_50_target: ">= 0.10"
  negative_sampling_quality: ">= 80% diverse negatives"
  category_coverage: ">= 90% of product categories"
  temporal_conservation: "maintain chronological order"
```

### **Dataset Specifications**

#### **Contrastive Learning Pairs**
```python
pair_generation:
  strategy: "Multi-stage negative sampling"

  positive_pairs:
    definition: "user-item interactions from transactions"
    min_interactions_per_user: 5
    max_interactions_per_user: 1000
    temporal_weighting: "recent interactions weighted higher"

  negative_sampling:
    random_negatives:
      count: 10
      per_positive: true
      sampling_method: "uniform from item corpus"

    in_batch_negatives:
      automatic: true
      method: "other items in current batch"
      expected_count: "batch_size - 1"

    hard_negatives:
      count: 5
      per_positive: true
      source: "item-item collaborative filtering"
      similarity_threshold: "> 0.3"
      diversity_constraint: "different categories preferred"
```

#### **Sequence Construction**
```python
sequence_requirements:
  max_sequence_length: 50
  padding_strategy: "post_padding"
  truncation: "longest_sequences_first"

  user_sequences:
    chronological: true
    include_session_boundaries: true
    min_sequence_length: 3
    max_gap_hours: 72

  temporal_features:
    recency_encoding: "exponential decay"
    day_of_week_encoding: "sinusoidal"
    seasonal_encoding: "cyclical"
```

#### **Category Balancing**
```python
balancing_strategy:
  category_weights:
    method: "inverse_frequency"
    smoothing_factor: 0.5
    min_weight: 0.1
    max_weight: 5.0

  user_segment_balancing:
    segments: ["new_users", "active_users", "churned_users"]
    target_distribution: [0.2, 0.6, 0.2]
    min_samples_per_segment: 1000
```

## ⚙️ **Generation Pipeline**

### **Phase 1: User-Item Interaction Matrix**
```python
interaction_matrix:
  sparse_format: "CSR matrix"
  dimensions: [num_users, num_items]
  value_encoding: "implicit feedback (1.0 for interactions)"

  preprocessing:
    filter_cold_users: "min 5 interactions"
    filter_cold_items: "min 10 interactions"
    temporal_validation: "train/test split by time"

  augmentation:
    add_view_data: "if available"
    weight_by_price: "log(price + 1)"
    channel_weighting: "online=1.2, offline=1.0"
```

### **Phase 2: Item Similarity Computation**
```python
item_similarity:
  algorithm: "Item-Item Collaborative Filtering"

  computation:
    similarity_metric: "cosine similarity"
    min_common_users: 5
    top_k_similar: 1000

  optimization:
    sparse_computations: true
    approximate_neighbors: true
    batch_size: 10000

  quality_control:
    diversity_threshold: 0.7  # max category similarity
    popularity_correction: "inverse document frequency"
```

### **Phase 3: Hard Negative Mining**
```python
hard_negative_mining:
  strategy: "multi-source negative candidates"

  candidate_sources:
    item_cf_similar:
      top_k: 100
      similarity_range: [0.3, 0.8]

    category_crossing:
      different_category: true
      same_price_range: true

    collaborative_candidates:
      similar_users_purchased: true
      temporal_proximity: "within 7 days"

  negative_scoring:
    difficulty_score: "weighted combination of similarities"
    diversity_penalty: "encourage diverse negatives"
    popularity_bias_correction: "down-weight popular items"
```

### **Phase 4: Dataset Construction**
```python
dataset_construction:
  structure: "PyTorch Dataset compatible"

  sample_format:
    user_features: "dict of tensors"
    item_features: "dict of tensors"
    label: "float32 (1.0 for positive, 0.0 for negative)"
    metadata: "dict with timestamps, sources"

  memory_optimization:
    lazy_loading: true
    background_prefetching: true
    compression: "zstd level 3"
    caching: "LRU cache for frequently accessed samples"
```

## 📤 **Output Specifications**

### **Training Files**

| File | Format | Compression | Splits | Size Estimate |
|------|--------|-------------|--------|---------------|
| `train_dataset.parquet` | Parquet | ZSTD | Train (80%) | ~8GB |
| `val_dataset.parquet` | Parquet | ZSTD | Validation (10%) | ~1GB |
| `test_dataset.parquet` | Parquet | ZSTD | Test (10%) | ~1GB |
| `item_similarity_matrix.parquet` | Parquet | Snappy | Single | ~200MB |
| `user_sequences.parquet` | Parquet | ZSTD | Single | ~500MB |
| `vocabulary.json` | JSON | None | Single | ~50KB |
| `category_mappings.json` | JSON | None | Single | ~20KB |

### **Dataset Schema**

#### **Training Sample Schema**
```python
training_sample_schema = {
    # User Features
    'user_id': 'string',
    'user_age': 'int16',
    'user_age_group': 'category',
    'user_club_status': 'category',
    'user_news_frequency': 'category',
    'user_lifetime_days': 'int16',
    'user_total_purchases': 'int32',
    'user_avg_basket_size': 'float32',
    'user_preferred_channel': 'int8',
    'user_interaction_sequence': 'list[int]',  # item_ids
    'user_sequence_lengths': 'int16',

    # Item Features
    'item_id': 'string',
    'item_product_group': 'category',
    'item_color_group': 'category',
    'item_garment_group': 'category',
    'item_department': 'category',
    'item_index_group': 'category',
    'item_popularity_score': 'float32',
    'item_price_category': 'category',
    'item_price_normalized': 'float32',

    # Interaction Features
    'label': 'float32',  # 1.0 for positive, 0.0 for negative
    'timestamp': 'datetime64[ns]',
    'sales_channel': 'int8',
    'day_of_week': 'int8',
    'month': 'int8',
    'season': 'category',
    'negative_source': 'category',  # random, hard, in_batch
    'similarity_score': 'float32',  # for hard negatives

    # Metadata
    'sample_id': 'string',
    'generation_timestamp': 'datetime64[ns]'
}
```

#### **Vocabulary Mappings**
```python
vocabulary_schema = {
    'user_features': {
        'age_groups': ['16-24', '25-34', '35-44', '45-54', '55+'],
        'club_status': ['ACTIVE', 'PRE-CREATION', 'LEFT CLUB'],
        'news_frequency': ['NONE', 'Regularly', 'Monthly', 'None']
    },
    'item_features': {
        'product_groups': ['Garment Upper body', 'Garment Lower body', 'Accessories', ...],
        'color_groups': ['Black', 'White', 'Blue', 'Red', ...],
        'garment_groups': ['Jersey Basic', 'Under-, Nightwear', ...]
    },
    'interaction_features': {
        'negative_sources': ['random', 'hard_negative', 'in_batch'],
        'seasons': ['Spring', 'Summer', 'Fall', 'Winter']
    }
}
```

## ✅ **Quality Gates & Validation**

### **Dataset Quality Checks**
```python
dataset_validation:
  structural_integrity:
    - no_missing_labels: "100% complete"
    - valid_user_item_pairs: "100% reference integrity"
    - proper_data_types: "100% schema compliance"
    - reasonable_value_ranges: "validated against domain knowledge"

  statistical_properties:
    - positive_negative_ratio: "1:11 minimum"
    - category_distribution: "similar to source data"
    - temporal_coverage: "maintain original time distribution"
    - user_item_coverage: "> 80% of users, > 90% of items"

  machine_learning_readiness:
    - feature_cardinality: "reasonable for embedding layers"
    - class_balance: "category weights applied correctly"
    - sequence_lengths: "appropriate distribution"
    - negative_quality: "diversity metrics > 0.7"
```

### **Train/Val/Test Split Validation**
```python
split_validation:
  temporal_splitting:
    train_period: "earliest 80% of time"
    val_period: "next 10% of time"
    test_period: "latest 10% of time"

  user_leakage_prevention:
    no_user_overlap: "users unique across splits"
    item_overlap_allowed: "items can appear in multiple splits"

  distribution_consistency:
    category_distribution: "KS test p > 0.05"
    user_activity_distribution: "similar across splits"
    price_distribution: "maintain original characteristics"
```

### **Performance Requirements**
```python
generation_performance:
  throughput:
    - minimum_rate: "5000 samples/second"
    - target_rate: "10000 samples/second"

  resource_usage:
    - max_memory_usage: "16GB"
    - max_cpu_usage: "80%"
    - max_disk_io: "200MB/second"

  timing:
    - total_generation_time: "< 30 minutes"
    - similarity_computation: "< 10 minutes"
    - negative_sampling: "< 15 minutes"
    - dataset_construction: "< 5 minutes"
```

## 🎛️ **Configuration & Hyperparameters**

### **Dataset Generation Parameters**
```python
config_parameters:
  negative_sampling:
    random_negatives_per_positive: 10
    hard_negatives_per_positive: 5
    in_batch_negatives: true
    similarity_threshold: 0.3

  sequence_processing:
    max_sequence_length: 50
    min_sequence_length: 3
    temporal_gap_hours: 72

  category_balancing:
    enable_balancing: true
    balance_threshold: 0.2  # max deviation from target
    smoothing_factor: 0.5

  filtering:
    min_user_interactions: 5
    min_item_interactions: 10
    max_user_interactions: 1000
```

### **Scalability Configuration**
```python
scalability_config:
  local_development:
    num_workers: 4
    batch_size: 10000
    chunk_size: 1000

  cloud_processing:
    num_workers: 32
    batch_size: 50000
    chunk_size: 5000
    distributed_processing: true

  memory_optimization:
    chunked_processing: true
    lazy_loading: true
    sparse_matrices: true
    compression_level: 3
```

## 📊 **Monitoring & Metrics**

### **Generation Metrics**
```python
monitoring_metrics:
  data_quality:
    - completeness_score: "percentage of non-null values"
    - consistency_score: "schema and business rule compliance"
    - diversity_score: "negative sample diversity"
    - coverage_score: "user and item coverage percentages"

  performance:
    - generation_rate: "samples per second"
    - memory_usage: "peak memory consumption"
    - disk_io_rate: "read/write throughput"
    - cpu_utilization: "percentage CPU usage"

  statistical:
    - label_distribution: "positive/negative ratio"
    - category_distribution: "across all categorical features"
    - temporal_distribution: "time-based patterns"
    - user_activity_distribution: "interaction frequency"
```

### **Alerting Thresholds**
```python
alerting:
  critical_alerts:
    - data_quality_score: "< 90%"
    - generation_failure: "any exception"
    - resource_exhaustion: "memory > 90% or disk full"

  warning_alerts:
    - performance_regression: "generation_rate < 5000/sec"
    - category_imbalance: "deviation > 30% from target"
    - low_negative_diversity: "diversity_score < 0.6"

  info_notifications:
    - generation_completion: "success notification"
    - milestone_reached: "50%, 75% completion"
```

## 🔧 **Technical Implementation**

### **Core Technologies**
```python
technology_stack:
  data_processing:
    primary: "Polars for high-performance data manipulation"
    secondary: "Pandas for compatibility"
    distributed: "Dask for large-scale processing"

  similarity_computation:
    library: "scikit-learn NearestNeighbors"
    optimization: "FAISS for approximate nearest neighbors"
    storage: "SciPy sparse matrices"

  storage:
    format: "Apache Parquet with ZSTD compression"
    partitioning: "column-based for efficient querying"
    indexing: "appropriate indexes for common queries"
```

### **Code Structure**
```python
implementation_structure:
  modules:
    - dataset_generator: "main orchestration"
    - negative_sampler: "negative sampling strategies"
    - similarity_computer: "item-item similarity"
    - sequence_builder: "user interaction sequences"
    - quality_validator: "data quality checks"

  interfaces:
    - DatasetConfig: "configuration management"
    - QualityMetrics: "quality assessment"
    - ProgressTracker: "generation monitoring"
```

---

## 📋 **Implementation Checklist**

### **Pre-Generation**
- [ ] Validate input processed datasets
- [ ] Verify sufficient disk space (>50GB)
- [ ] Check system requirements (16GB RAM, 8+ cores)
- [ ] Validate configuration parameters

### **During Generation**
- [ ] Monitor memory usage and performance
- [ ] Validate intermediate results
- [ ] Track generation progress
- [ ] Log quality metrics

### **Post-Generation**
- [ ] Validate output datasets against quality gates
- [ ] Verify schema compliance
- [ ] Check train/val/test split integrity
- [ ] Generate dataset statistics report
- [ ] Archive generation logs and metadata

### **Documentation**
- [ ] Document configuration parameters
- [ ] Create dataset statistics report
- [ ] Update data catalog
- [ ] Archive generation code and version

---

**Contract Status**: ✅ Active
**Next Review**: 2025-11-17
**Implementation Owner**: ML Engineering Team
**Stakeholders**: Data Science Team, MLOps Team, Research Team