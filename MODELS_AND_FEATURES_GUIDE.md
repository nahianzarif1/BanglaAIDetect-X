# Models and Features - Complete Guide

## Current Architecture

### Currently Used Model: TF-IDF + Stylometry + Logistic Regression
**Location**: `src/models/baseline.py`

This is the model currently in use. It does NOT use BERT.

### Available but Not Used: BanglaBERT
**Location**: `src/models/banglabert.py`

This BERT-based model exists but is not being used in the current pipeline.

---

## Model 1: Baseline Model (Currently Active)

### Location
**File**: `src/models/baseline.py`

### Architecture
```
Input Text → [Features] → [Classifier] → Prediction
            ↓              ↓
    TF-IDF (word)   Logistic Regression
    TF-IDF (char)   (with class weighting)
    Stylometry (22 features)
```

### How It Works (Code Explanation)

#### 1. Feature Extraction (Lines 59-80)
```python
union = FeatureUnion([
    ("word", TfidfVectorizer(
        analyzer="word",              # Word-level features
        ngram_range=(1, 2),          # Unigrams and bigrams
        max_features=12000,          # Top 12,000 words
        min_df=2,                    # Must appear in 2+ documents
        sublinear_tf=True,           # TF normalization
    )),
    ("char", TfidfVectorizer(
        analyzer="char_wb",          # Character-level features
        ngram_range=(3, 5),          # 3-5 character n-grams
        max_features=15000,          # Top 15,000 character patterns
        min_df=2,
        sublinear_tf=True,
    )),
    ("style", Pipeline([
        ("extract", StylometryTransformer()),  # 22 stylometric features
        ("scale", StandardScaler()),           # Normalize features
    ])),
])
```

#### 2. Classifier (Lines 81-87)
```python
clf = LogisticRegression(
    C=0.8,                          # Regularization strength
    max_iter=2000,                  # Maximum iterations
    class_weight="balanced",        # Handle class imbalance
    solver="liblinear",             # Efficient for smaller datasets
    random_state=42,                # Reproducibility
)
```

#### 3. Training (Lines 91-95)
```python
def train_baseline(train_df: pd.DataFrame, cfg: dict):
    x_train, y_train = to_xy(train_df, cfg)  # Extract features and labels
    pipe = build_pipeline(cfg)                 # Build the pipeline
    pipe.fit(x_train, y_train)                 # Train the model
    return pipe
```

### Features Used (Total: ~27,000 features)

#### A. Word TF-IDF (~12,000 features)
- **What**: Word frequency vectors
- **How**: `TfidfVectorizer(analyzer="word")`
- **N-grams**: Unigrams and bigrams (1-2 word sequences)
- **Examples**: "ঢাকা", "বাংলাদেশ", "শিক্ষা", "উন্নয়ন"

#### B. Character TF-IDF (~15,000 features)
- **What**: Character pattern vectors
- **How**: `TfidfVectorizer(analyzer="char_wb")`
- **N-grams**: 3-5 character sequences
- **Examples**: "ঢাক", "কা", "শিক", "উন্ন"

#### C. Stylometric Features (22 features)
**Location**: `src/features/stylometry.py`

##### 1. Sentence Structure (5 features)
```python
"n_tokens": Total word count
"n_sentences": Total sentence count
"avg_sent_len": Average sentence length
"std_sent_len": Sentence length variance
"burstiness": Sentence length coefficient of variation
```

##### 2. Lexical Diversity (4 features)
```python
"type_token_ratio": Unique words / total words
"hapax_ratio": Words appearing exactly once / total words
"avg_word_len": Average word length
"std_word_len": Word length variance
```

##### 3. Content Features (4 features)
```python
"digit_ratio": Digit character density
"year_count_norm": Year pattern density per sentence
"punct_ratio": Punctuation character density
"paren_ratio": Parenthesis density
```

##### 4. Discourse Markers (3 features)
```python
"ai_marker_density": AI-style connectors (এছাড়াও, অন্যদিকে, সর্বোপরি)
"human_marker_density": Human markers (যদিও, তবে, অনুযায়ী)
"marker_balance": AI marker density - Human marker density
```

##### 5. Structural Patterns (6 features)
```python
"repeated_bigram_ratio": Repeated word pairs
"start_formulaic": Template opening patterns
"end_formulaic": Template closing patterns
"parallel_comma_lists": "X, Y এবং Z" patterns
"unique_punct_types": Variety of punctuation used
"comma_ratio": Comma density per sentence
```

### How Features Are Extracted (Code)

#### Stylometry Extraction (Lines 80-140 in stylometry.py)
```python
def extract_style_features(text: str) -> dict:
    text = text or ""
    sents = _sentences(text)                    # Split into sentences
    words = _words(text)                        # Split into words
    
    # Calculate sentence features
    sent_lens = [len(_words(s)) for s in sents]
    avg_sl = float(sent_lens.mean())
    std_sl = float(sent_lens.std())
    burstiness = std_sl / avg_sl if avg_sl > 0 else 0.0
    
    # Calculate lexical diversity
    counts = Counter(words)
    types = len(counts)
    hapax = sum(1 for _, c in counts.items() if c == 1)
    
    # Count discourse markers
    ai_m = _count_markers(text, AI_MARKERS)
    hu_m = _count_markers(text, HUMAN_MARKERS)
    
    # Calculate content features
    digit_ratio = len(_DIGIT.findall(text)) / max(len(text), 1)
    year_count_norm = len(_YEAR.findall(text)) / n_sent
    
    return {
        "n_tokens": float(len(words)),
        "n_sentences": float(len(sents)),
        "avg_sent_len": avg_sl,
        "std_sent_len": std_sl,
        "burstiness": burstiness,
        "type_token_ratio": types / n_tok,
        "hapax_ratio": hapax / n_tok,
        # ... all 22 features
    }
```

### Performance on Your Dataset
- **Validation**: 99.4% F1 Score
- **Test**: 99.2% F1 Score
- **ROC-AUC**: 99.98%

---

## Model 2: BanglaBERT (Available but Not Used)

### Location
**File**: `src/models/banglabert.py`

### Architecture
```
Input Text → [BanglaBERT Tokenizer] → [BanglaBERT Model] → [Classification Head] → Prediction
```

### How It Works (Code Explanation)

#### 1. Model Loading (Lines 76-78)
```python
model_name = "csebuetnlp/banglabert"  # Pre-trained BanglaBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2  # Binary classification (human vs AI)
)
```

#### 2. Data Preparation (Lines 37-57)
```python
def load_datasets(cfg: dict, tokenizer):
    train_df = pd.read_csv(cfg["paths"]["train"])
    
    def to_ds(df: pd.DataFrame) -> Dataset:
        df["labels"] = (df[cfg["dataset"]["label_col"]] == "ai").astype(int)
        ds = Dataset.from_pandas(df[[cfg["dataset"]["text_col"], "labels"]])
        
        def tokenize(batch):
            return tokenizer(
                batch[cfg["dataset"]["text_col"]],
                truncation=True,
                max_length=256,              # Maximum sequence length
                padding="max_length",        # Pad to max length
            )
        
        return ds.map(tokenize, batched=True)
    
    return to_ds(train_df), to_ds(val_df), to_ds(test_df)
```

#### 3. Training Configuration (Lines 85-99)
```python
args = TrainingArguments(
    output_dir="models/banglabert_detector",
    per_device_train_batch_size=8,        # Batch size
    learning_rate=2e-5,                  # Learning rate
    num_train_epochs=5,                   # Number of epochs
    warmup_ratio=0.1,                    # Warmup ratio
    weight_decay=0.01,                   # Weight decay
    eval_strategy="epoch",               # Evaluate every epoch
    save_strategy="epoch",               # Save every epoch
    load_best_model_at_end=True,         # Keep best model
    metric_for_best_model="f1",          # Optimization metric
)
```

#### 4. Training (Lines 101-110)
```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,     # F1, precision, recall, ROC-AUC
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()  # Fine-tune BanglaBERT
```

### Features Used (Transformer-based)
BanglaBERT uses **learned embeddings** rather than hand-crafted features:

#### A. Token Embeddings
- Each Bangla word is converted to a vector
- Learned from large Bangla corpus training
- Captures semantic relationships between words

#### B. Positional Embeddings
- Encodes word position in sentence
- Allows model to understand word order

#### C. Attention Mechanism
- Focuses on relevant parts of the text
- Learns which words are important for classification

#### D. Contextual Representations
- Each word's representation depends on context
- Captures nuanced linguistic patterns

### Why BanglaBERT is Better (Potentially)
1. **Contextual Understanding**: Understands word context, not just frequency
2. **Semantic Learning**: Captures meaning, not just patterns
3. **Transfer Learning**: Pre-trained on large Bangla corpus
4. **Attention Mechanism**: Can focus on important text regions

### Why BanglaBERT is Not Currently Used
1. **Computational Cost**: Requires GPU for training
2. **Training Time**: Slower than TF-IDF (hours vs minutes)
3. **Memory Usage**: Requires more memory
4. **Current Performance**: TF-IDF already achieves 99%+ on your dataset

---

## Model 3: Fusion Model (Similar to Baseline)

### Location
**File**: `src/models/fusion.py`

### Architecture
Similar to baseline but with additional configuration options. Currently not actively used in the pipeline.

---

## How to Use BanglaBERT Instead

### Step 1: Install Required Dependencies
```bash
pip install torch transformers datasets
```

### Step 2: Update Pipeline to Use BanglaBERT
Modify `run_pipeline.py` to include BanglaBERT training:

```python
# After baseline training
if run_command("python -m src.models.banglabert", "BanglaBERT training"):
    print("BanglaBERT training completed")
```

### Step 3: Update Inference to Use BanglaBERT
Modify `src/inference.py` to load BanglaBERT model instead of baseline.

### Step 4: Expected Performance
BanglaBERT may improve performance on:
- Unseen topics
- Different writing styles
- Shorter texts
- More nuanced linguistic patterns

---

## Feature Comparison

| Feature Type | Baseline (TF-IDF) | BanglaBERT |
|--------------|-------------------|------------|
| Word Frequency | Yes (TF-IDF) | Yes (Embeddings) |
| Character Patterns | Yes (TF-IDF) | No |
| Sentence Structure | Yes (Hand-crafted) | Yes (Learned) |
| Discourse Markers | Yes (Hand-crafted) | Yes (Learned) |
| Context | No | Yes (Attention) |
| Semantic Meaning | No | Yes (Embeddings) |
| Training Speed | Fast (minutes) | Slow (hours) |
| Inference Speed | Fast | Moderate |
| Memory Usage | Low | High |

---

## Current Performance Summary

### Baseline Model (Currently Active)
- **Features**: ~27,000 (12K word TF-IDF + 15K char TF-IDF + 22 stylometric)
- **Accuracy**: 99.2% F1 on test set
- **Training Time**: ~5 minutes
- **Model Size**: ~50MB

### BanglaBERT (Available but Not Used)
- **Features**: Learned embeddings ( contextual)
- **Expected Accuracy**: Potentially 95-99% (varies by dataset)
- **Training Time**: ~2-4 hours (with GPU)
- **Model Size**: ~400MB

---

## Recommendation

For your current use case with 5,999 samples:

1. **Keep Baseline**: Already achieving 99%+ accuracy
2. **Consider BanglaBERT**: If you need better generalization to unseen topics
3. **Ensemble Approach**: Combine both models for potentially better performance

The baseline model is working excellently on your dataset. BanglaBERT would be useful if you need to handle more diverse writing styles or improve performance on completely unseen topics.