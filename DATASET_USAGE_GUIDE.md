# Dataset Usage Guide and Performance Explanation

## How to Use Your Excel Dataset

Your Excel file `[Bangla]_AI_&_HUMAN_Text_For_Classification.xlsx` has been successfully integrated into the project.

### Dataset Details
- **Total samples**: 5,999 rows
- **AI samples**: 2,999 (Gemini, DeepSeek, ChatGPT, Grok, GPT)
- **Human samples**: 3,000 (Wikipedia, Ittefaq news)
- **Columns**: source_text, Source, Source Name

### How It Was Integrated
1. The Excel file was copied to: `data/raw/ieee_dataport_dataset.xlsx`
2. Created `load_custom_dataset.py` to convert Excel format to project schema
3. Mapped sources:
   - `AI` → label: `ai`, generator: `ai`
   - `ittefaq` → label: `human`, generator: `human`
   - `wiki` → label: `human`, generator: `human`
4. Created topic-based grouping for proper train/validation/test splitting

### Training Results with Your Dataset
```
Validation Set (894 samples):
- Precision: 99.3%
- Recall: 99.6%
- F1 Score: 99.4%
- ROC-AUC: 99.99%

Test Set (898 samples):
- Precision: 98.9%
- Recall: 99.6%
- F1 Score: 99.2%
- ROC-AUC: 99.98%
```

**Note**: These are excellent results on your dataset!

## Explaining Performance Issues to Your Teacher

### Why the Model Works Well on Training Data but Struggles on Test Data

#### 1. **Topic Bias (Most Likely Issue)**
**Problem**: The model may be learning topics rather than writing style.

**Explanation to Teacher**:
- "Our model is trained on specific topics from the dataset (e.g., environment, education, agriculture)
- When tested on the same or similar topics, performance is excellent (99%+ accuracy)
- When tested on completely new topics not seen during training, the model may struggle
- This is a common issue in AI detection models - they can learn topic-specific patterns instead of general writing style differences"

**Solution**:
- Use a truly held-out test set with completely different topics
- Ensure train/validation/test splits are topic-based (which we do)
- Cross-validate on multiple topic groupings

#### 2. **Dataset Distribution Mismatch**
**Problem**: Training data distribution differs from real-world test data.

**Explanation to Teacher**:
- "The training dataset contains specific AI models (Gemini, DeepSeek, ChatGPT, Grok, GPT-5.3)
- If test data uses different AI models (e.g., Claude, Llama, local models), performance may drop
- Similarly, human writing styles in training (Wikipedia, Ittefaq) may differ from student writing
- The model learns patterns specific to the training distribution"

**Solution**:
- Test on diverse AI models not in training
- Include student-written samples in training
- Use domain adaptation techniques

#### 3. **Style Transfer and Paraphrasing**
**Problem**: AI can mimic human writing style through paraphrasing.

**Explanation to Teacher**:
- "Modern AI models (GPT-4, Claude) can be instructed to write in specific styles
- Students can ask AI to 'write like a student' or 'make it less formal'
- This reduces the stylometric differences our model relies on
- This is a fundamental challenge in AI detection - the boundary between human and AI writing is blurring"

**Solution**:
- Include paraphrased AI samples in training
- Use more advanced features (semantic patterns, attribution)
- Acknowledge this limitation honestly

#### 4. **Length and Context Factors**
**Problem**: Short texts or different contexts are harder to classify.

**Explanation to Teacher**:
- "The model works best with longer texts (40+ words) because there are more style features
- Short texts provide insufficient stylometric signals
- Different contexts (academic papers vs. casual writing) have different style patterns
- Our model was trained on specific genres (news, encyclopedia content)"

**Solution**:
- Train on diverse genres and contexts
- Use ensemble methods for different text lengths
- Provide confidence scores for short texts

### Honest Assessment for Your Teacher

**What to say:**

"Sir, our model achieves 99%+ accuracy on the test set from our training dataset, which contains 5,999 samples with balanced AI and human text. However, we need to be careful about overinterpreting these results:

1. **Topic Bias**: The model may be learning topic-specific patterns rather than general writing style differences. Our topic-based splitting helps, but more diverse topic coverage is needed.

2. **Model Generalization**: The training data uses specific AI models (Gemini, DeepSeek, ChatGPT, Grok, GPT-5.3). Performance may vary with different AI models or newer versions.

3. **Style Mimicry**: Modern AI can be instructed to write in specific styles, potentially evading detection. This is a fundamental challenge in AI detection.

4. **Dataset Limitations**: Our human samples come from Wikipedia and Ittefaq news, which may differ from student writing styles.

**Current Performance**: 
- Excellent on dataset splits (99%+ F1)
- Real-world performance likely lower due to these factors
- This is a research prototype, not a production-grade detection system

**Recommendations for Improvement**:
1. Include student-written samples in training
2. Test on unseen AI models
3. Add paraphrased AI samples
4. Use ensemble methods with multiple base models
5. Implement continuous learning for new AI patterns"

### Technical Details for Documentation

**Methodology**:
- Feature fusion: TF-IDF (word + character) + 22 stylometric features
- Model: Logistic regression with class weighting
- Validation: Topic-based 70/15/15 split to prevent data leakage
- Calibration: Isotonic regression for probability estimation

**Features Used**:
- Lexical diversity (type-token ratio, hapax legomena)
- Sentence structure (burstiness, length variance)
- Discourse markers (AI vs human connectors)
- Content features (digit density, year patterns)
- Structural patterns (formulaic openings, parallel lists)

**Limitations Acknowledged**:
- Topic bias in training data
- Limited to specific AI models in training
- Struggles with style-mimicking AI
- Genre-specific performance variations
- Length sensitivity (works best with 40+ words)

## Running with Your Dataset

The system is now configured to use your Excel file automatically:

```bash
# The pipeline will automatically detect and use your Excel file
python run_pipeline.py

# Run the app
streamlit run app/app.py
```

## Future Improvements

1. **Expand Training Data**: Add more diverse AI models and human writing styles
2. **Cross-Validation**: Use k-fold validation with different topic groupings
3. **Ensemble Methods**: Combine multiple detection approaches
4. **Continuous Learning**: Update model with new AI patterns
5. **Explainability**: Add feature importance analysis for debugging

This honest assessment shows academic integrity while acknowledging the technical challenges in AI detection.