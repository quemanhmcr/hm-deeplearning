# Data Contract 3: Two-Tower Model Architecture

## 🔍 **Contract Overview**

**Contract ID**: DC-003
**Version": 1.0.0
**Owner**: ML Research Team
**Status**: Active
**Created**: 2025-11-10
**Review Date**: 2025-11-17
**Dependencies**: DC-002 (Training Dataset Generation)

## 🏗️ **Model Architecture Overview**

### **Core Design Principles**
```python
design_principles:
  scalability: "support millions of users and items"
  efficiency: "sub-10ms inference latency"
  maintainability: "modular, testable architecture"
  extensibility: "easy to add new features"
  production_ready: "optimized for serving"
```

### **Architecture Type**
```python
model_type: "Two-Tower Neural Network"
paradigm: "Contrastive Learning with Hard Negatives"
embedding_space: "Joint 128-dimensional space"
training_objective: "Multi-phase training with different loss functions"
serving_pattern: "Independent tower serving with vector similarity"
```

## 🧠 **Model Architecture Specification**

### **User Tower Architecture**
```python
user_tower:
  input_dim: 128  # After feature preprocessing
  output_dim: 128  # Final embedding dimension
  embedding_layers:

    # Categorical Feature Embeddings
    categorical_embeddings:
      club_member_status:
        vocab_size: 5
        embedding_dim: 8
        pretrained: false

      fashion_news_frequency:
        vocab_size: 5
        embedding_dim: 8
        pretrained: false

      age_group:
        vocab_size: 6
        embedding_dim: 16
        pretrained: false

      preferred_channel:
        vocab_size: 3
        embedding_dim: 4
        pretrained: false

    # Numerical Feature Processing
    numerical_features:
      - name: "age"
        normalization: "min_max_scaling"
        learned_embedding: true
        embedding_dim: 8

      - name: "lifetime_days"
        normalization: "log_transform + min_max"
        learned_embedding: true
        embedding_dim: 8

      - name: "total_purchases"
        normalization: "log_transform + min_max"
        learned_embedding: true
        embedding_dim: 8

      - name: "avg_basket_size"
        normalization: "standard_scaling"
        learned_embedding: true
        embedding_dim: 8

    # Sequence Processing (Interaction History)
    sequence_encoder:
      type: "Transformer Encoder"
      max_sequence_length: 50
      d_model: 64
      num_heads: 4
      num_layers: 2
      dropout: 0.1

      # Positional Encoding
      positional_encoding:
        type: "learned"
        max_len: 50

    # Dense Layers
    dense_layers:
      - layer_1:
          units: 256
          activation: "ReLU"
          dropout: 0.15
          batch_norm: true

      - layer_2:
          units: 128
          activation: "ReLU"
          dropout: 0.15
          batch_norm: true

      - layer_3:
          units: 64
          activation: "ReLU"
          dropout: 0.1
          batch_norm: true

    # Final Projection
    projection_head:
      layers:
        - dense_1:
            units: 128
            activation: "ReLU"
            dropout: 0.1
        - dense_2:
            units: 128
            activation: "none"  # Final embedding
```

### **Item Tower Architecture**
```python
item_tower:
  input_dim: 256  # After feature preprocessing
  output_dim: 128  # Final embedding dimension
  embedding_layers:

    # Categorical Feature Embeddings
    categorical_embeddings:
      product_group_name:
        vocab_size: 20
        embedding_dim: 16
        pretrained: false

      color_group_name:
        vocab_size: 50
        embedding_dim: 16
        pretrained: false

      garment_group_name:
        vocab_size: 50
        embedding_dim: 16
        pretrained: false

      department_name:
        vocab_size: 30
        embedding_dim: 12
        pretrained: false

      index_group_name:
        vocab_size: 10
        embedding_dim: 8
        pretrained: false

      section_name:
        vocab_size: 25
        embedding_dim: 12
        pretrained: false

      price_category:
        vocab_size: 10
        embedding_dim: 8
        pretrained: false

    # Text Feature Processing
    text_features:
      - name: "product_name"
        embedding_type: "pretrained"
        model: "sentence-transformers/all-MiniLM-L6-v2"
        embedding_dim: 384
        freeze_pretrained: true
        projection_dim: 64

      - name: "detail_desc"
        embedding_type: "pretrained"
        model: "sentence-transformers/all-MiniLM-L6-v2"
        embedding_dim: 384
        freeze_pretrained: true
        projection_dim: 64

    # Numerical Features
    numerical_features:
      - name: "price_normalized"
        normalization: "as_is"  # already normalized 0-1
        learned_embedding: false
      - name: "popularity_score"
        normalization: "min_max_scaling"
        learned_embedding: true
        embedding_dim: 8

    # Dense Layers
    dense_layers:
      - layer_1:
          units: 512
          activation: "ReLU"
          dropout: 0.15
          batch_norm: true

      - layer_2:
          units: 256
          activation: "ReLU"
          dropout: 0.15
          batch_norm: true

      - layer_3:
          units: 128
          activation: "ReLU"
          dropout: 0.1
          batch_norm: true

    # Final Projection
    projection_head:
      layers:
        - dense_1:
            units: 128
            activation: "ReLU"
            dropout: 0.1
        - dense_2:
            units: 128
            activation: "none"  # Final embedding
```

## 🎯 **Training Objectives & Loss Functions**

### **Phase 1: Cold Start Embedding Convergence**
```python
phase_1_config:
  name: "Cold Start Embedding Convergence"
  epochs: 2-3
  loss_function: "BPR Loss + L2 Regularization"

  bpr_loss:
    type: "Bayesian Personalized Ranking"
    margin: 0.1
    sampling_strategy: "random_negatives_only"

  regularization:
    l2_weight_decay: 1e-5
    dropout_rate: 0.1
    gradient_clipping: 1.0

  optimization:
    optimizer: "AdamW"
    learning_rate: 1e-3  # Constant
    scheduler: "constant"
    weight_decay: 1e-5

  batch_size: 2048
  mixed_precision: false  # More stable for cold start
```

### **Phase 2: Large-Batch Contrastive Learning**
```python
phase_2_config:
  name: "Large-Batch Contrastive Learning"
  epochs: 3-5
  loss_function: "InfoNCE with Temperature Scaling"

  infonce_loss:
    type: "Information Noise Contrastive Estimation"
    temperature: 0.08
    label_smoothing: 0.1

  sampling_strategy:
    in_batch_negatives: true
    additional_random_negatives: 10
    popularity_correction: "inverse_frequency_weighting"
    popularity_weight: 0.5

  optimization:
    optimizer: "AdamW"
    learning_rate: 0.032  # Cosine decay to 5e-4
    scheduler: "cosine_annealing_warm_restarts"
    weight_decay: 1e-5

  training_optimization:
    batch_size: 8192
    gradient_accumulation: 4  # Effective batch size: 32K
    mixed_precision: true  # FP16 for speed and memory
    gradient_clipping: 1.0
```

### **Phase 3: Hard Negative Mining & Fine-tuning**
```python
phase_3_config:
  name: "Hard Negative Mining & Fine-tuning"
  epochs: 2-3
  loss_function: "Multi-objective: InfoNCE + Category Alignment"

  composite_loss:
    primary_loss:
      type: "InfoNCE"
      weight: 0.9
      temperature: 0.08

    auxiliary_loss:
      type: "Category Alignment Loss"
      weight: 0.1  # α = 0.1
      temperature: 0.1

    total_loss: "L_total = λ₁ × L_InfoNCE + λ₂ × L_category"

  hard_negative_strategy:
    similarity_threshold: 0.3
    max_hard_negatives: 5
    diversity_constraint: "different categories preferred"

  optimization:
    optimizer: "AdamW"
    learning_rate: 5e-5  # Small constant LR for fine-tuning
    scheduler: "constant"
    weight_decay: 1e-5

  regularization:
    dropout_rate: 0.15
    label_smoothing: 0.05
    gradient_clipping: 0.5

  batch_size: 4096
  mixed_precision: true
```

### **Phase 4: Embedding Stabilization**
```python
phase_4_config:
  name: "Embedding Stabilization"
  epochs: 1
  loss_function: "Positive Pair Only + EMA Regularization"

  stabilization_loss:
    type: "Positive Pair Consistency"
    temperature: 0.05
    no_negative_sampling: true

  ema_regularization:
    decay_rate: 0.999  # EMA_decay = 0.999
    apply_to: "both_towers"
    update_frequency: "every_step"

  optimization:
    optimizer: "AdamW"
    learning_rate: 1e-5  # Linear decay
    scheduler: "linear_decay"
    weight_decay: 1e-6

  batch_size: 2048
  mixed_precision: true
  gradient_clipping: 0.1  # Very conservative
```

## 🔧 **Technical Implementation**

### **Model Class Structure**
```python
class TwoTowerModel(nn.Module):
    def __init__(self, config):
        self.user_tower = UserTower(config.user_tower)
        self.item_tower = ItemTower(config.item_tower)
        self.temperature = config.temperature
        self.ema_decay = config.ema_decay

    def forward(self, user_features, item_features, labels=None):
        user_embeddings = self.user_tower(user_features)
        item_embeddings = self.item_tower(item_features)

        if self.training:
            return self.compute_loss(user_embeddings, item_embeddings, labels)
        else:
            return user_embeddings, item_embeddings

    def compute_loss(self, user_emb, item_emb, labels):
        # Multi-phase loss computation
        pass

    def get_user_embeddings(self, user_features):
        return self.user_tower(user_features)

    def get_item_embeddings(self, item_features):
        return self.item_tower(item_features)
```

### **Training Pipeline Interface**
```python
class TrainingPipeline:
    def __init__(self, model_config, training_config):
        self.model = TwoTowerModel(model_config)
        self.optimizer = self.setup_optimizer(training_config)
        self.scheduler = self.setup_scheduler(training_config)
        self.scaler = torch.cuda.amp.GradScaler() if training_config.mixed_precision else None

    def train_epoch(self, dataloader, phase):
        for batch in dataloader:
            loss = self.training_step(batch, phase)
            self.backward_pass(loss, phase)

    def training_step(self, batch, phase):
        # Phase-specific forward pass
        pass

    def backward_pass(self, loss, phase):
        # Phase-specific optimization
        pass
```

## 📊 **Model Performance Specifications**

### **Inference Requirements**
```python
inference_requirements:
  latency:
    single_user_embedding: "< 5ms"
    single_item_embedding: "< 3ms"
    batch_inference_1k: "< 100ms"

  throughput:
    cpu_inference: "> 1000 predictions/second"
    gpu_inference: "> 10K predictions/second"

  memory:
    user_tower_memory: "< 200MB"
    item_tower_memory: "< 300MB"
    total_model_memory: "< 1GB"

  accuracy:
    recall_50: ">= 0.10"
    ndcg_10: ">= 0.15"
    coverage: ">= 80%"
    cold_start_performance: "< 20% degradation"
```

### **Scalability Requirements**
```python
scalability_requirements:
  user_scale: "10M+ users"
  item_scale: "100K+ items"
  embedding_storage: "< 5GB for all embeddings"
  update_frequency: "daily incremental updates"

  serving:
    horizontal_scaling: "multiple model replicas"
    caching: "Redis for popular embeddings"
    vector_search: "FAISS for similarity search"
```

## 🎛️ **Hyperparameter Configuration**

### **Global Hyperparameters**
```python
global_config:
  embedding_dim: 128
  temperature: 0.08
  dropout_rate: 0.15
  weight_decay: 1e-5
  gradient_clipping: 1.0
  mixed_precision: true
  use_ema: true

  optimization:
    optimizer: "AdamW"
    betas: [0.9, 0.999]
    eps: 1e-8
    amsgrad: false
```

### **Phase-Specific Hyperparameters**
```python
phase_hyperparameters:
  phase_1:
    learning_rate: 1e-3
    batch_size: 2048
    epochs: 2-3
    scheduler: "constant"

  phase_2:
    learning_rate: 0.032
    min_learning_rate: 5e-4
    batch_size: 8192
    epochs: 3-5
    scheduler: "cosine_annealing"

  phase_3:
    learning_rate: 5e-5
    batch_size: 4096
    epochs: 2-3
    scheduler: "constant"

  phase_4:
    learning_rate: 1e-5
    batch_size: 2048
    epochs: 1
    scheduler: "linear_decay"
```

## 📤 **Output Specifications**

### **Model Artifacts**
```python
output_artifacts:
  checkpoints:
    - path: "checkpoints/best_model.pt"
      content: "Complete model state dict"
      format: "PyTorch"

    - path: "checkpoints/user_tower.pt"
      content: "User tower only"
      format: "PyTorch"

    - path: "checkpoints/item_tower.pt"
      content: "Item tower only"
      format: "PyTorch"

    - path: "checkpoints/optimizer.pt"
      content: "Optimizer state"
      format: "PyTorch"

  deployment:
    - path: "artifacts/user_tower.onnx"
      content: "User tower optimized"
      format: "ONNX"
      optimization: "FP16"

    - path: "artifacts/item_tower.onnx"
      content: "Item tower optimized"
      format: "ONNX"
      optimization: "FP16"

    - path: "artifacts/model_config.json"
      content: "Complete configuration"
      format: "JSON"

    - path: "artifacts/embedding_mappings.json"
      content: "Feature to embedding mappings"
      format: "JSON"

  evaluation:
    - path: "metrics/training_metrics.json"
      content: "Training history and metrics"
      format: "JSON"

    - path: "metrics/evaluation_results.json"
      content: "Final evaluation metrics"
      format: "JSON"

    - path: "plots/training_curves.png"
      content: "Training and validation curves"
      format: "PNG"
```

### **Model Size Specifications**
```python
model_size_requirements:
  pytorch_model:
    user_tower: "< 100MB"
    item_tower: "< 150MB"
    complete_model: "< 300MB"

  onnx_optimized:
    user_tower: "< 50MB"
    item_tower: "< 75MB"
    total_optimized: "< 150MB"

  embedding_storage:
    user_embeddings: "num_users × 128 × 4 bytes"
    item_embeddings: "num_items × 128 × 4 bytes"
    vocab_embeddings: "< 50MB"
```

## ✅ **Quality Gates & Validation**

### **Model Quality Checks**
```python
model_validation:
  architecture_validation:
    - parameter_count: "within expected range"
    - gradient_flow: "no vanishing/exploding gradients"
    - embedding_dimensions: "correct output shapes"
    - loss_convergence: "decreasing loss trend"

  performance_validation:
    - training_speed: "> 1000 samples/second"
    - memory_usage: "< 90% GPU memory"
    - convergence_rate: "loss improves each epoch"
    - final_metrics: "meets minimum thresholds"

  deployment_validation:
    - onnx_conversion: "successful without errors"
    - inference_latency: "meets requirements"
    - model_size: "within specified limits"
    - numerical_stability: "consistent predictions"
```

### **Training Monitoring**
```python
training_monitoring:
  real_time_metrics:
    - training_loss: "per batch"
    - learning_rate: "per step"
    - gpu_utilization: "continuous"
    - memory_usage: "continuous"
    - gradient_norms: "per batch"

  epoch_metrics:
    - validation_loss: "per epoch"
    - recall_50: "per epoch"
    - ndcg_10: "per epoch"
    - coverage: "per epoch"
    - category_distribution: "per epoch"
```

---

## 📋 **Implementation Checklist**

### **Pre-Training**
- [ ] Validate training dataset schema
- [ ] Verify GPU availability and memory
- [ ] Initialize model with correct configuration
- [ ] Setup logging and monitoring
- [ ] Create backup directories

### **During Training**
- [ ] Monitor training progress and metrics
- [ ] Validate gradient flow and convergence
- [ ] Check memory usage and performance
- [ ] Save checkpoints at regular intervals
- [ ] Log all hyperparameters and configurations

### **Post-Training**
- [ ] Validate final model performance
- [ ] Convert to deployment formats (ONNX)
- [ ] Generate evaluation report
- [ ] Archive training logs and artifacts
- [ ] Update model registry

### **Documentation**
- [ ] Document all hyperparameters and results
- [ ] Create model card with capabilities and limitations
- [ ] Update inference serving documentation
- [ ] Archive training code and configuration

---

**Contract Status**: ✅ Active
**Next Review**: 2025-11-17
**Implementation Owner**: ML Research Team
**Stakeholders**: ML Engineering Team, MLOps Team, Product Team